# 求解转置卷积里outputpadding的函数.input_size、output_size为输入输出的特征图大小
def get_outputpadding(input_size, output_size, kernel_size, stride, padding):
    outputpadding = output_size - (input_size - 1) * stride + 2 * padding - kernel_size
    return outputpadding
