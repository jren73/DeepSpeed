import torch
from torch.autograd import Variable
import sys
import collections
import time


def get_variable_device(v):
    if torch.is_tensor(v):
        dev = v.get_device()
    else:
        dev = v[0].get_device()
    return dev


def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        obj = Variable(obj)
    if isinstance(obj, Variable):
        if dev == 'cpu':
            target = torch.empty_like(obj, device=dev).copy_(obj)
        else:
            target = obj.cuda(dev, async=True)

        if main_stream is not None:
            target.data.record_stream(main_stream)
        return target
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj


def main():
    p = []
    for i in range(10):
        #x=torch.rand(2+i,3+i).cuda()
        x = torch.rand(3 + i, 4 + i, device='cuda:1')
        p.append(x)
    length = sum([t.numel() for t in p])
    a = torch.ones(int(length), dtype=torch.half, device=torch.cuda.current_device())

    q = torch.zeros([length * 2], device='cpu')

    migration_stream = torch.cuda.Stream()
    main_stream = torch.cuda.current_stream()

    start = time.time()
    index = 0
    with torch.cuda.stream(migration_stream):
        for t in p:
            size = t.numel()
            q[index:index + size] = async_copy_to(t.view(-1), 'cpu', main_stream)
            index += size
        q[length:2 * length] = async_copy_to(a, 'cpu', main_stream)
    stop = time.time()
    print(q)
    print(get_variable_device(q))
    print(get_variable_device(p))
    print(get_variable_device(a))
    print("migration_time = ", stop - start)


if __name__ == "__main__":
    main()
    #x=torch.rand(3,2,dtype=torch.double)
    #print(x)
