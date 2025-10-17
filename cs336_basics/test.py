import multiprocessing
import os
import time

def square(x):
    time.sleep(0.1)
    return x*x

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        result = pool.map(square, range(100))
        print(f"result: {result}")

        result_chunk = pool.map(square, range(100), chunksize=10)
        print(f"result_chunk: {result_chunk}")