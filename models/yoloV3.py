import torch
import torch.nn as nn
from collections import OrderedDict
from models.backbone import darknet
from utils.config import TRAINING_PARAMS

backboneNet = {
    "darknet21" : darknet.darknet21,
    "darknet53" : darknet.darknet53,
}

class YoloNet(nn.Module):
    def __init__(self, config):
        super(YoloNet, self).__init__()

        self.config = config
        self.model_params = config["model_params"]      # backbone_name, backbone_pretrained_weights

        # backbone
        backbone_func = backboneNet[self.model_params["backbone_name"]]     #选择 yolov3
        self.backbone = backbone_func(self.model_params["backbone_pretrained"]) # 传入参数
        _out_filters = self.backbone.layers_out_filters     # 导出 输出

        # embadding 0   416 ⬇ 32 = 13*13
        # 对于每个像素点都将产生 3 * 85 = 255 维向量：
        #  3 anchors * (4坐标值 + 1conf + 80类)
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)

        # embadding 1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.embedding1_cbl = self._make_cbl(512, 256, 1)       # route, layer85
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)

        #  embedding2
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.embedding2_cbl = self._make_cbl(256, 128, 1)       #   route, layer97
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):            # to generate the yolo layer 1 3 1 3 1 3
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))     # 1*1 prediction layer82, 94, 106
        return m

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, op in enumerate(_embedding):      # 前向传递 embedding 层
                _in = op(_in)
                if i == 4:          # 第四个的时候，需要新分支一个route路
                    out_branch = _in
            return _in, out_branch

        x2, x1, x0 = self.backbone(x)

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)

        # yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)

        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        return out0, out1, out2

    def load_darknet_weights(self, weights_path):
        import numpy as np
        #Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values
        # Needed to write header when saving weights
        weights = np.fromfile(fp, dtype=np.float32)         # The rest are weights
        print ("total len weights = ", weights.shape)
        fp.close()

        ptr = 0
        all_dict = self.state_dict()
        all_keys = self.state_dict().keys()
        print (all_keys)
        last_bn_weight = None
        last_conv = None
        for i, (k, v) in enumerate(all_dict.items()):
            if 'bn' in k:
                if 'weight' in k:
                    last_bn_weight = v
                elif 'bias' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_bias: ", ptr, num_b, k)
                    ptr += num_b
                    # weight
                    v = last_bn_weight
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_weight: ", ptr, num_b, k)
                    ptr += num_b
                    last_bn_weight = None
                elif 'running_mean' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_mean: ", ptr, num_b, k)
                    ptr += num_b
                elif 'running_var' in k:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("bn_var: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
                else:
                    raise Exception("Error for bn")
            elif 'conv' in k:
                if 'weight' in k:
                    last_conv = v
                else:
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv bias: ", ptr, num_b, k)
                    ptr += num_b
                    # conv
                    v = last_conv
                    num_b = v.numel()
                    vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
                    v.copy_(vv)
                    print ("conv wight: ", ptr, num_b, k)
                    ptr += num_b
                    last_conv = None
        print("Total ptr = ", ptr)
        print("real size = ", weights.shape)



if __name__ == "__main__":
    net = YoloNet(TRAINING_PARAMS)
    inp = torch.randn(1, 3, 416, 416)

    out0, out1, out2 = net(inp)

    print(out0[0].shape)
    print(out1.size())
    print(out2.size())