
import torch
from thop import clever_format, profile
import time

from nets.FastDETR import DETR

if __name__ == '__main__':
    input_shape         = [640, 640]
    # ---------------------------------------------#
    #   resnet18
    #   VAN
    #   cspdarknet
    #   mobilenetone
    #   mobilenetv2
    #   mobilenetv3
    # ---------------------------------------------#
    backbone            = "VAN"
    num_classes         = 5
    aux_loss            = True
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = DETR( backbone, num_classes=num_classes,  aux_loss=aux_loss)
    for i in model.children():
        print(i)
        print('==============================')
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)


    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

    iterations = None
    input = torch.randn(1, 3, 640, 640).cuda()

    model = model.eval()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    print(latency)

