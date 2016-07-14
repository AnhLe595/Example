import tensorflow as tf
import numpy as np
import pandas as pd
import time


#Parameters
learning_rate = 0.01
training_epochs = 2
display_step = 1
					#Dataset
# FOLDER = 'Linear_Model/'

TRAINING_BASE =  'ftrain_2000.csv'
TRAINING_PREDICT =  'ftest_2000.csv'
TRAIN = 'train_1000.csv'
# TEST_DATA = FOLDER + 'test_1000.csv'
					#Load Data
# training_base = tf.contrib.learn.datasets.base.load_csv(filename = TRAINING_BASE, target_dtype = np.int)
# # training_base = tf.contrib.learn.datasets.base.load_csv(filename = TRAINING_BASE, target_dtype = np.int)
# training_predict = tf.contrib.learn.datasets.base.load_csv(filename=TRAINING_PREDICT, target_dtype=np.int)
# test = tf.contrib.learn.datasets.base.load_csv(filename=TEST_DATA, target_dtype=np.in
#########################################################################################################################################
#												using pandas to load csv => It works													#

training_base = pd.read_csv(TRAINING_BASE)
training_predict = pd.read_csv(TRAINING_PREDICT)
train = pd.read_csv(TRAIN)

############################################################################################################

# global variables
client_product_median = dict()
product_median = dict()



#build statistics from training data
def train_data(train_file):
    f = open(train_file, "r")
    f.readline()

    agency_client_product_demand = dict()
    client_product_demand = dict()
    product_demand = dict()
    client_demand = dict()
    total_row = 0
    count = 0

    client_id, product_id, demand = 0, 0, 0

    while 1:
        line = f.readline().strip()
        if line == '':
            break
        total_row += 1
        if total_row % 5000000 == 0:
            print('Read {} lines...'.format(total_row))
        if train_file == TRAINING_BASE:
        	arr = line.split(",")
        	week_index  = int(arr[0]) - 3
        	client_id   = arr[4]
        	product_id  = arr[5]
        	demand      = int(arr[10])
        	
        else:
        	arr = line.split(",")
        	week_index  = int(arr[1]) - 3
        	client_id   = arr[2]
        	product_id  = arr[3]
        	demand      = int(arr[4])
        	
        # if total_row > 1e6:
        #     break
        
        #obtain information
        # arr = line.split(",")
        # week_index  = int(arr[0]) - 3
        # client_id   = arr[4]
        # product_id  = arr[5]
        # demand      = int(arr[10])
	    

	    
	  #   arr = line.split(",")
	  #   if train_file == TRAINING_BASE:
	  #   	# client_id = arr[4]
	  #   	# product_id = arr[5]
	  #   	# demand = int(arr[10])
			# print "Pass"
	  #   # else:
	  #   # 	client_id = arr[5]
	  #   # 	product_id = arr[6]
	  #   # 	demand = int(arr[11])

        #built dictionaries
        f1 = (client_id, product_id)
        if f1 in client_product_demand:
            client_product_demand[f1].append(demand)
        else:
            client_product_demand[f1] = [demand]

        f3 = (product_id)
        if f3 in product_demand:
            product_demand[f3].append(demand)
        else:
            product_demand[f3] = [demand]
    f.close()
    print("Medians calculating...")
    for x, y in client_product_demand.items():
        client_product_median[x] = np.median(y)
    for x, y in product_demand.items():
        product_median[x] = np.median(y)

    print("Done training, total rows %d" %(total_row))
    return client_product_median, product_median
def col_to_arr(data, trans_file):
	X,Y = [],[]
	for i,j in data.items():
		if trans_file == TRAINING_BASE:
			X.append(j)
		else: 
			Y.append(j)
	return X,Y

x,_ = train_data(TRAINING_BASE)
y,_ = train_data(TRAINING_PREDICT)
train_base_X,_ = col_to_arr(x, TRAINING_BASE)
_,train_predict_Y = col_to_arr(y,TRAINING_PREDICT)

# print train_predict_Y


#																																		#
# #########################################################################################################################################
# 					#Assign variable in feature and targer
# # train_base_feature, training_predict_target = training_base.data, training_predict.target
# # train_base_X = train_base_feature['Demanda_uni_equil']
# # train_predict_Y = training_predict_target['Demanda_uni_equil']

# 					#How to take a value of weight and bias 
# # Y' = X*W + B
# # Y': Value of prediction , X: from TRAINING_DATA 
# # W: is a slope , W = r*sX/sY 				r: correlation between X and Y
# #											sX: standard deviation of X
# #											sY: stadard deviation of Y
# # B: Y intercept , A = My - B*Mx			My: mean Y
# #											Mx: mean X

# start_time = time.time()
# corr = np.corrcoef(train_base_X,train_predict_Y)
# std_X = np.std(train_base_X)

# std_Y = np.std(train_predict_Y)
# val_weight = corr * std_Y / std_X


# mean_X = np.mean(train_base_X)
# mean_Y = np.mean(train_predict_Y)
# val_b = mean_Y - val_weight * mean_X



# #array obtain Demanda_uni_equil
# train_base_X, train_predict_Y = [],[]
# for i in range(1000):
# 	train_base_X.append(training_base['Demanda_uni_equil'][i])
# 	train_predict_Y.append(training_predict['Demanda_uni_equil'][i])
# print len(train_base_X)

#########################################################################################################################################

length_of_train = len(train_base_X)
start_time = time.time()
					#Create variable place holder
hold_var_X = tf.placeholder('float')
hold_var_Y = tf.placeholder('float')


					#Create variable
W = tf.Variable(np.random.randn(),name='weight')
b = tf.Variable(np.random.randn(),name='bias')

					#Prediction by linear model
predict = tf.add(tf.mul(hold_var_X,W), b)

					#Mean square error
cost = tf.reduce_sum(tf.pow(predict-hold_var_Y,2))/(2*length_of_train)

#Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initializing the variables
init = tf.initialize_all_variables()

count = 0
summ = 0
# #launch the graph
with tf.Session() as sess:
	sess.run(init)
	#Fit all training data
	# for epoch in range(training_epochs):
	for (x,y) in zip(train_base_X,train_predict_Y):
		sess.run(optimizer, feed_dict={hold_var_X:x, hold_var_Y:y})
	# 		#Display logs per epoch step
	# if (epoch+1) % display_step == 0:
	c = sess.run(cost, feed_dict={hold_var_X:x, hold_var_Y:y})
	print "cost=", "{:.9f}".format(c),"W=", sess.run(W), "b=", sess.run(b)

	print 'Optimization Finished!'
	print "Eslapse time=", time.time() - start_time
	training_cost = sess.run(cost, feed_dict={hold_var_X:train_base_X, hold_var_Y:train_predict_Y})
	print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'
# # ####################################################TESTING########################################################

