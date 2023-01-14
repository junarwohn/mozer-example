from threading import Thread
import time
from multiprocessing import Process, Queue

# def work(id, start, end, result):
#     total = 0
#     for i in range(start, end):
#         total += i
#     result.append(total)
#     return

# One thread
"""
if __name__ == "__main__":
    START, END = 0, 100000000
    result = list()
    th1 = Thread(target=work, args=(1, START, END, result))
    
    stime = time.time()
    th1.start()
    th1.join()
    etime = time.time()
    running_time = etime - stime
"""

# # Multi thread - FUCK GIL
# if __name__ == "__main__":
#     START, END = 0, 100000000
#     result = list()
#     # th1 = Thread(target=work, args=(1, START, END, result))
#     th1 = Thread(target=work, args=(1, START, END//2, result))
#     th2 = Thread(target=work, args=(2, END//2, END, result))
    
#     stime = time.time()
#     th1.start()
#     th2.start()
#     th1.join()
#     th2.join()
#     etime = time.time()
#     running_time = etime - stime



    

# Multi Process
def work(id, start, end, result):
    total = 0
    for i in range(start, end):
        total += i
    result.put(total)
    return 0

if __name__ == "__main__":
    START, END = 0, 100000000
    result = Queue()
    th1 = Process(target=work, args=(1, START, END//2, result))
    th2 = Process(target=work, args=(2, END//2, END, result))
    
    stime = time.time()
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    etime = time.time()
    running_time = etime - stime

    result.put('STOP')
    total = 0
    while True:
        tmp = result.get()
        if tmp == 'STOP':
            break
        else:
            total += tmp
    print(f"Result: {total}")


print(f"running time {running_time}")