import tensorflow as tf

def f(x):
    return x**2

def calculus(func, a, b):
    epsilon=0.001
    width=abs(a-b)
    seg=width/epsilon
    area=0.0
    for i in range(int(seg)):
        subarea=epsilon*(f(a)+f(b))/2
        area += subarea

    return area

print("area is:", calculus(f,6,9))

tf.app.flags.DEFINE_boolean("mytest",False, "True unless indicated")
FLAGS=tf.app.flags.FLAGS

def main(argv=None):
    if(FLAGS.mytest):
        print("this is just a self test")
    else:
        print("nothing happens")

if __name__ == "__main__":
    tf.app.run()