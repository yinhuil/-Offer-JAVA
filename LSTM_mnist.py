import tensorflow as tf
from tensorflow.contrib import rnn
from learnDemo import input_data

mnist=input_data.read_data_sets("K:/deepLearning/mnist",one_hot=True)

#  超参数
lr=0.001
training_iters=1000000
batch_size=128

n_input=28
n_step=28
n_hidden_unis=128
n_class=10

x=tf.placeholder(tf.float32,[None,n_step,n_input])
y=tf.placeholder(tf.float32,[None,n_class])

weights={
    'in':tf.Variable(tf.random_normal([n_input,n_hidden_unis])),
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_class]))
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_class,]))
}
def RNN(X,weights,biases):
#     hidden layer for input to cell
    X=tf.reshape(X,[-1,n_input])
    X_in=tf.matmul(X,weights['in'])+biases['in']
    X_in=tf.reshape(X_in,[-1,n_step,n_hidden_unis])

    lstm_cell=rnn.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
    result=tf.matmul(states[1],weights['out'])+biases['out']
    return result


pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size<training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_step,n_input])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step%20==0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
            }))
        step+=1
