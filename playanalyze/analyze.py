from playanalyze.models import *
from playanalyze.helper import *
from playanalyze.labels import *
import playanalyze.ml as ml
import math
import inspect as inspect
import numpy as np
import pylab as pl
import math
from matplotlib.colors import ListedColormap
from sklearn import neighbors, cross_validation,datasets, svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from prettytable import PrettyTable
import random as random


#####################################
####### Global Parameters ###########
#####################################

# Play Labels
PLAYS = {26:'ELBOW',42:'FLOPPY',53:'HORNS',92:'INVERT',98:'DELAY',51:'DROP',7:'DRAG',9:'POSTUP',501:'RANDOM',0:'NOT PLAY',1:'PLAY'}

# I created the manual times when each play starts for Variable Length Segmenetation
MANUAL_TIMES = {26:ELBOW_TIMES,42:FLOPPY_TIMES,53:HORNS_TIMES,92:INVERT_TIMES,98:DELAY_TIMES,51:DROP_TIMES,7:DRAG_TIMES,9:POSTUP_TIMES,501:RANDOM_TIMES}
# The typical length of each play for variable length segmentation, I've messed around with this a bunch
LENGTHS = {26:3,42:4,92:2,53:2,98:4,9:2,501:6}
# Court segementation parameters, which is used to generate the areas of the court (see: generateDegreeDistanceList(court_dic,points))
STANDARD_COURT_DD = {10:[90],18:[45,90,135,180],25:[45,90,135,180],100:[45,90,135,180]}

#####################################
####### Generate Base Positions #####
#####################################
# I make this it's own function, because it takes quite a bit of time to load in all of the possession 
# data. This function, if you are uploading all the plays, takes ~30 minutes on a new macbook pro.

# This function returns a list of each play with it's easy_positions
def basePositions():
	dic = {'RANDOM':RANDOM_FILTERED,'ELBOW':ELBOW,'FLOPPY':FLOPPY,'HORNS':HORNS,'INVERT':INVERT,'DELAY':DELAY,'POSTUP':POSTUP}
	for key,value in dic.iteritems():
		relevant_possession_ids = value
		possessions = Possession.objects.filter(id__in=relevant_possession_ids)
		for possession in possessions:
			try:
				possession.positions = getPositions(possession)
				possession.play_id = [keys for keys,val in PLAYS.items() if val==key][0] # Reverse assign play ids in case they are mislabeled (which many are)
			except:
				print possession.id
		print str(key)+" complete"
		dic[key] = list(possessions)
	ALL = list(dic['FLOPPY'])+list(dic['INVERT'])+list(dic['HORNS'])+list(dic['ELBOW'])+list(dic['DELAY'])+list(dic['RANDOM'])+list(dic['POSTUP'])
	return ALL

########################################################################
####### Compares the different styles 6 second vs variable second} #####
####################'###################################################

# This function is a speed optimization. When running variable length segmenation,
# we create subsets of each possession, this function makes all of those combinations
# so we only have to run it once as opposed to every iteration of the ML algorithms
def pp(basePositions):
	# Variables
	FULL_LENGTH_TIMES = [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
	ALL_LENGTHS = [value for key,value in LENGTHS.items()]
	
	# Generate all positive / negative vectors to speed up parameter optimization
	basePositions = [basePosition for basePosition in basePositions if basePosition.positions['play_start']-basePosition.positions['play_end'] > 2 and (basePosition.id not in [9439,6326,5724,4447])]
	vectors = {}
	for bp in basePositions:
		dict = {}
		dict['play_id'] = bp.play_id
		play_id = bp.play_id
		if play_id != 501:
			# Generate the exact positive windowed vector
			start_time = MANUAL_TIMES[play_id][bp.id]
			play_length = LENGTHS[bp.play_id]
			window = convertgetPositionToCustomTime(bp.positions,start_time,start_time-play_length)
			window_positive = buildVector(window,timeListFromPlayLength(play_length),False)
			dict['window_positive'] = window_positive
		
		# Generate windowed vectors of all lengths
		window_negatives_list = {}
		for AL in ALL_LENGTHS:
			window_possessions = windowPositionsFromGetPositions(bp.positions,AL,.5)
			window_negatives = [buildVector(positions,timeListFromPlayLength(AL),False) for positions in window_possessions]
			window_negatives_list[AL] = window_negatives
		dict['window_negatives'] = window_negatives_list

		# Generate the full length vector
		full_length = buildVector(bp.positions,FULL_LENGTH_TIMES,False)
		dict['full_length'] = full_length

		# Add them to the dictionary
		vectors[bp.id] = dict

	return vectors

########### The function to run a specific method ##############
## type = can be ocw = one vs. all with fixed length segmenation
#                ocfl = one. vs. all with variable length segmentation
#                mcfl = multi-class with fixed length segmentation
#                mcw = multi-class with variable length segmentation
# basePositions = the output of basePositions()
# op = the output of pp(basePositions), which is an optimization step
def runDifferent(type,basePositions,op):
	PLAY_TO_TEST = play_id
	# 1. Shuffle and filter out bad examples (less than 2 seconds long)
	random.shuffle(basePositions)
	basePositions = [basePosition for basePosition in basePositions if basePosition.positions['play_start']-basePosition.positions['play_end'] > 2 and (basePosition.id not in [9439,6326,5724,4447])]
	# 2. Stratified split of the examples
	labels = [basePosition.play_id for basePosition in basePositions]
	skf = cross_validation.StratifiedKFold(labels,5)
	# 3. Iterate through each stratified training example
	if type == 'ocw':
		return oneClassWindowedML(basePositions,skf,PLAY_TO_TEST,C,class_weight,op)
	elif type == 'ocfl':
		return oneClassFullLengthML(basePositions,skf,PLAY_TO_TEST,C,class_weight,op)
	elif type == 'mcfl':
		return multiClassFullLengthML(basePositions,skf,op)
	elif type == 'mcw':
		return multiClassWindowedML(basePositions,skf,op,C)

def oneClassFullLengthML(basePositions,skf,play_to_test,C,class_weight,op):
	results = []
	for train_index,test_index in skf:
		# Generate the SVMs from the training examples
		train_set = retrieveIndexes(basePositions,list(train_index))
		training_vectors = [op[tsi]['full_length'] for tsi in [bp.id for bp in train_set]]
		training_labels = [(1 if play_to_test == possession.play_id else 0) for possession in train_set]
		
		# Generate the SVM
		clf = svm.SVC(C=C,class_weight={1:class_weight,0:1},kernel='rbf',probability=True)
		clf.fit(training_vectors, training_labels)

		# Test them 
		test_set = retrieveIndexes(basePositions,list(test_index))
		test_vectors = [op[tsi]['full_length'] for tsi in [bp.id for bp in test_set]]
		test_labels = [(1 if play_to_test == possession.play_id else 0) for possession in test_set]
		test_ids = [possession.id for possession in test_set]
		
		for index,test_point in enumerate(test_vectors):
			results.append((clf.predict(test_point).tolist()[0],test_labels[index],test_ids[index]))
	return results

########### Scores the Full Length ML Using OVA Strategy ##############
def multiClassFullLengthML(basePositions,skf,op):
	TIMES = [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
	results = []
	for train_index,test_index in skf:
		# Generate the SVMs from the training examples
		train_set = retrieveIndexes(basePositions,list(train_index))
		training_vectors = [op[tsi]['full_length'] for tsi in [bp.id for bp in train_set]]
		training_labels = [possession.play_id for possession in train_set]
		training_labels_set = list(set(training_labels))
		if 501 in training_labels_set: # Remove 501 from the svm generation if its part of the training_labels
			training_labels_set.remove(501)

		# Generate the SVMs
		svms = {}
		for label in training_labels_set:
			unit_training_labels = [(1 if label==val else 0) for val in training_labels] # Convert labels to 1's and 0's
			params = {26:(2,4),53:(2,9),92:(1,9),98:(3,2),9:(2,5),42:(1,4)}
			clf = svm.SVC(C=params[label][0],class_weight={1:params[label][1],0:1},kernel='rbf',probability=True)
			clf.fit(training_vectors, unit_training_labels)
			svms[label] = clf

		# Test them 
		test_set = retrieveIndexes(basePositions,list(test_index))
		test_vectors = [op[tsi]['full_length'] for tsi in [bp.id for bp in test_set]]
		test_labels = [possession.play_id for possession in test_set]
		test_ids = [possession.id for possession in test_set]
		for index,test_point in enumerate(test_vectors):
			highest_probability = 0
			highest_probability_value = 0
			for key,value in svms.items():
				probability = svms[key].predict_proba(test_point)
				probability = probability[0].tolist()[1]
				highest_probability = key if highest_probability_value < probability else highest_probability
				highest_probability_value = max(probability,highest_probability_value)
			if highest_probability_value > .3:
				results.append((highest_probability,test_labels[index],test_ids[index]))
			else:
				if 501 in list(set(training_labels)):
					results.append((501,test_labels[index],test_ids[index]))
				else:
					results.append((highest_probability,test_labels[index],test_ids[index]))

	print '\nFull Length Confusion Matrix Performance'
	#print resultGeneration(results)
	return results

def oneClassWindowedML(basePositions,skf,play_to_test,C,class_weight,op):
	results = []
	for train_index,test_index in skf:
		# Generate the SVMs from the training examples
		train_set = retrieveIndexes(basePositions,list(train_index))
		test_set = retrieveIndexes(basePositions,list(test_index))

		# Generate the positive vectors
		bp_ids = [bp.id for bp in train_set]
		length = LENGTHS[play_to_test]
		positive_training_vectors = [op[bp_id]['window_positive'] for bp_id in bp_ids if op[bp_id]['play_id']==play_to_test]
		positive_training_labels = [1 for i in positive_training_vectors]
		ntv = [op[bp_id]['window_negatives'][length] for bp_id in bp_ids if op[bp_id]['play_id']!=play_to_test]
		negative_training_vectors = []
		for a in ntv:
			for b in a:
				negative_training_vectors.append(b)
		negative_training_labels = [0 for i in negative_training_vectors]

		# Combine the vectors
		training_vectors = positive_training_vectors+negative_training_vectors
		training_labels = positive_training_labels+negative_training_labels
		
		# Generate the SVM
		clf = svm.SVC(C=C,class_weight={1:class_weight,0:1},kernel='rbf',probability=True)
		clf.fit(training_vectors, training_labels)
		
		# Test them 
		for possession in test_set:
			true_label = 1 if possession.play_id == play_to_test else 0
			window_possessions = op[possession.id]['window_negatives'][length]
			guess = 0
			for wp in window_possessions:
				window_guess = list(clf.predict(wp))[0]
				guess = 1 if (guess or window_guess) else 0
			results.append((guess,true_label,possession.id))
	return results


########### This is multiclass scoring of the windowed method with OVA ##############
def multiClassWindowedML(basePositions,skf,op,C):
	# Labels
	all_labels = [possession.play_id for possession in basePositions]
 	labels = list(set(all_labels)) # List of unique play IDs (minus random)
 	if 501 in labels:
 		labels.remove(501)

 	results = [] # Stores the results in the form of tuples (guess,actual_label)
 	ranked_results = []
	for train_index,test_index in skf:

		# Generate the SVMs from the training examples
		train_set = retrieveIndexes(basePositions,list(train_index))
		test_set = retrieveIndexes(basePositions,list(test_index))

		svm_dictionary = {}
		for play_to_test in labels:
			# Generate the positive vectors
			bp_ids = [bp.id for bp in train_set]
			length = LENGTHS[play_to_test]
			positive_training_vectors = [op[bp_id]['window_positive'] for bp_id in bp_ids if op[bp_id]['play_id']==play_to_test]
			positive_training_labels = [1 for i in positive_training_vectors]
			ntv = [op[bp_id]['window_negatives'][length] for bp_id in bp_ids if op[bp_id]['play_id']!=play_to_test]
			negative_training_vectors = []
			for a in ntv:
				for b in a:
					negative_training_vectors.append(b)
			negative_training_labels = [0 for i in negative_training_vectors]

			# Combine the vectors
			training_vectors = positive_training_vectors+negative_training_vectors
			training_labels = positive_training_labels+negative_training_labels
			
			# Generate the SVM
			params = {26:(1,6),53:(1,4),92:(1,6),98:(3,2),9:(1,9),42:(1,4)}
			#params = {26:(1,25),53:(1,8),92:(1,13),98:(1,12),9:(1,7),42:(2,7)}
			#params[play_to_test][0]
			clf = svm.SVC(C=params[play_to_test][0],class_weight={1:params[play_to_test][1],0:1},kernel='rbf',probability=True)
			clf.fit(training_vectors, training_labels)
			svm_dictionary[play_to_test] = clf # Store the fitted SVM in the dictionary

		# 3. Test the holdout set
		for possession in test_set:
			# Run each test point through each SVM
			poss_result = [] # Track any positives
			for key,clf in svm_dictionary.items():
				play_length = LENGTHS[key]
				window_possessions = op[possession.id]['window_negatives'][play_length]
				highest_probability = 0
				for poss in window_possessions:
					probability = clf.predict_proba(poss)[0].tolist()[1]
					highest_probability = max(highest_probability,probability)
				poss_result.append((highest_probability,key))
			# Look at all the positives and probabilities, to determine the ultimate guess
			poss_result = sorted(poss_result,reverse=True)
			if 501 in all_labels:
				ultimate_guess = 501 if poss_result[0][0] < .1 else poss_result[0][1]
			else:
				ultimate_guess = poss_result[0][1]
			# Store the Actual and the Guess
			results.append((ultimate_guess,possession.play_id,possession.id))
			ranked_results.append((rankResults(poss_result),possession.play_id,possession.id))
	print 'Multi Class Windowed Performance'
	print resultGeneration(results)
	print scoreRanked(ranked_results)
	return results

def rankResults(probabilities):
	prob = sorted(probabilities,reverse=True)
	final_predict = []
	added_501 = False
	for p in prob:
		if p[0] < .1 and not added_501:
			final_predict.append(501)
			added_501 = True
		final_predict.append(p[1])
	return final_predict

def scoreRanked(list_of_results):
	for i in list_of_results:
		print i
	total = [0,0,0,0,0,0,0]
	for i in range(len(list_of_results)):
		for j in range(len(list_of_results[i][0])):
			if list_of_results[i][0][j] == list_of_results[i][1]:
				total[j] += 1
	return total

############# Takes list of tuples (guess,actual) and generates F scores and confusion matrices ##########
def resultGeneration(list_of_results):
	## Print wrong instances
	# for result in list_of_results:
	# 	if result[0] != result[1]:
	# 		print 'Guess: '+str(PLAYS[result[0]])+" Actual: "+str(PLAYS[result[1]])+" | ID: "+str(result[2])
	## Print Correct Classification Rate
	score = sum([(1 if label==guess else 0) for guess,label,id in list_of_results])/float(len(list_of_results))
	print 'Correct Classification Rate: '+str(round(score,2))
	## Compute the F Score
	f1score = f1_score([label for guess,label,id in list_of_results], [guess for guess,label,id in list_of_results], average='macro')
	print 'Micro F1 Score: '+str(round(f1score,2))
	# Compute the Confusion Matrix
	labels = sorted(list(set([label for guess,label,id in list_of_results])))
	cm = [[0]*len(labels) for lb in range(len(labels))]
	counter = 0
	for result in list_of_results:
		guess = labels.index(result[0])
		actual = labels.index(result[1])
		cm[actual][guess] += 1
	plays = [PLAYS[label] for label in labels]
	x = PrettyTable(['']+plays)
	for i in range(len(cm)):
		x.add_row([PLAYS[labels[i]]]+cm[i])
	print x

def arrayf1score(array):
	length = int(len(array)**.5)
	results = []
	for a in range(len(array)):
		row = int(math.floor(a/length))
		col = a % length
		for i in range(array[a]):
			results.append((row,col,None))
	print CCR(results)
	return f1Score(results)


def f1Score(list_of_results):
	labels = [label for guess,label,id in list_of_results]
	guesses = [guess for guess,label,id in list_of_results]
	#print precision_recall_fscore_support(labels,guesses,beta=2,average="micro")[2]
	return f1_score(labels,guesses,average="weighted")

def CCR(list_of_results):
	return sum([(1 if label==guess else 0) for guess,label,id in list_of_results])/float(len(list_of_results))

def CM(list_of_results):
	labels = sorted(list(set([label for guess,label,id in list_of_results])))
	cm = [[0]*len(labels) for lb in range(len(labels))]
	counter = 0
	for result in list_of_results:
		guess = labels.index(result[0])
		actual = labels.index(result[1])
		cm[actual][guess] += 1
	return cm

########### This is the one vs all algorithm with windowed algorithm ##############
def scoreWindowedML(basePositions,skf,PLAY_TO_TEST):
	training_confusion = [[0,0],[0,0]]
	confusion = [[0,0],[0,0]]
	correct,total = 0,0
	TIMES = [0,1,2]
	pass_1,pass_2 = [0,0,0,0,0,0],[0,0,0,0,0,0]
	grr = []
	for train_index,test_index in skf:
		# 4. Generate training and test sets
		train_set = retrieveIndexes(basePositions,list(train_index))
		test_set = retrieveIndexes(basePositions,list(test_index))
		# Generate window_possessions from sliding window scale
		training_vectors = []
		for possession in train_set:
			if possession.play_id == PLAY_TO_TEST:
				if possession.id in POSTUP_TIMES.keys():
					training_vectors.append((convertgetPositionToCustomTime(possession.positions,POSTUP_TIMES[possession.id],POSTUP_TIMES[possession.id]-2),1,possession.id))
					#training_vectors.append((convertgetPositionToCustomTime(possession.positions,POSTUP_TIMES[possession.id]+.5,POSTUP_TIMES[possession.id]-1.5),1,possession.id))
					#training_vectors.append((convertgetPositionToCustomTime(possession.positions,POSTUP_TIMES[possession.id]-.5,POSTUP_TIMES[possession.id]-2.5),1,possession.id))
			else:
				window_possessions = windowPositionsFromGetPositions(possession.positions,2,.5)
				for positions in window_possessions:
					training_vectors.append((positions,0,possession.id))
		# Convert position objects into machine learning vectors
		ml_vectors = []
		ml_labels = []
		ml_id = []
		for positions,label,id in training_vectors:
			printed = True if label==1 else False
			vector = buildVector(positions,TIMES,False)
			ml_vectors.append(vector)
			ml_labels.append(label)
			ml_id.append(id)
			# if label == 1:
			# 	pass_1 = [pass_1[i]+vector[i] for i in range(len(vector))]
			# 	if vector[3] == 0:
			# 		print buildVector(positions,TIMES,True)
			# 		print str(vector)+": "+str(label)+"  -  "+str(id)
			# 		grr.append(id)
			# 		print '-----------\n'
			# else:
			# 	pass_2 = [pass_2[i]+vector[i] for i in range(len(vector))]
		# Run the machine learning algorithms
		clf = svm.SVC(C=.3,class_weight={1:8,0:1},kernel='linear',probability=True)
		clf.fit(ml_vectors, ml_labels)

		# How dows the training set do? #
		for possession in train_set:
			label = 1 if possession.play_id == PLAY_TO_TEST else 0
			window_possessions = windowPositionsFromGetPositions(possession.positions,2,.5)
			final_label = 0
			instances = []
			for positions in window_possessions:
				vector = buildVector(positions,TIMES,False)
				guess = list(clf.predict(vector))[0]
				instances.append((positions['play_start'],positions['play_end'],guess))
				final_label = 1 if (guess or final_label == 1) else 0
			training_confusion[(1 if label==1 else 0)][(1 if final_label==1 else 0)] += 1

		# Test the holdout set #
		pos_in_set = 0
		for possession in test_set:
			label = 1 if possession.play_id == PLAY_TO_TEST else 0
			pos_in_set += label
			window_possessions = windowPositionsFromGetPositions(possession.positions,2,.5)
			final_label = 0
			instances = []
			for positions in window_possessions:
				vector = buildVector(positions,TIMES,False)
				guess = list(clf.predict(vector))[0]
				instances.append(guess)
				final_label = 1 if (guess or final_label == 1) else 0
			# if not (final_label == 0 and label == 0):
			# 		print '\n'
			# 		print possession.id
			# 		print 'Predicted Label: '+str(final_label)
			# 		print 'Actual Label: '+str(label)
			# 		print instances
			confusion[(1 if label== 1 else 0)][(1 if final_label==1 else 0)] += 1
			correct += 1 if label==final_label else 0
			total += 1
	print pass_1
	print pass_2
	print set(grr)
	# Print Out All The Final Scores
 	print '-------------------------------'
 	print 'Training Confusion Matrix'
 	ml.prettyPrintConfusionMatrix(training_confusion,[0,1])
 	print 'Actual Confusion Matrix'
 	ml.prettyPrintConfusionMatrix(confusion,[0,1])
 	print 'Final Score: '+str(float(correct)/total)
 	print '\n'
 	return float(correct)/total


############# generate time vector based on the length of the play ##########
def timeListFromPlayLength(length):
	INCREMENT = 1
	return [i*INCREMENT for i in range(int(length/INCREMENT)+1)]

########### Constructs the vectors used in the algorithms ##############
def buildVector(positions,TIMES,printed):
	positions = strongSidePositions(positions) # canocalize the possession
	bv = buildPassVectors(positions,TIMES,printed)+buildClosenessVectors(positions,TIMES)+buildDDPositionVectors(positions,TIMES)+buildUniqueMeasureVectors(positions,TIMES)
	return bv

#####################################
#### Strong/Weak Side Function ######
#####################################
# This flips every play based on the ball position so all plays run to the bottom left side of the court

def strongSidePositions(positions):
	# Find the average ball y value during the possession
	new_positions = positions.copy()
	sum_ball_y, count_ball_y = 0, 0
	for eps in new_positions['easy_positions']:
		sum_ball_y += eps['ball'][1]
		count_ball_y += 1
	average_y = sum_ball_y / count_ball_y
	if average_y > 25:
		new_easy_positions = []
		for ep in new_positions['easy_positions']:
			new_easy_positions.append(flipEasyPositionInstance(ep))
		new_positions['easy_positions'] = new_easy_positions
	return new_positions

def flipEasyPositionInstance(easy_position):
	new_team_positions = [(x,50-y) for x,y in easy_position['team_positions']]
	new_defense_positions = [(x,50-y) for x,y in easy_position['defense_positions']]
	new_ball_positions = (easy_position['ball'][0],50-easy_position['ball'][1],easy_position['ball'][2])
	easy_position['team_positions'] = new_team_positions
	easy_position['defense_positions'] = new_defense_positions
	easy_position['ball'] = new_ball_positions
	return easy_position


#####################################
####### Play Subset (Postup) ########
#####################################
def subsetRunGridSearch(basePositions):
	# Grid Search Criteria
	C_array = [5]
	class_weights = [3]
	max_score = 0
	scores = []
	for C in C_array:
		row = []
		for class_weight in class_weights:
			score = subsetRun(basePositions,C,class_weight)
			max_score = max(max_score,score)
			row.append(round(score,2))
		scores.append(row)
	print 'Grid Search: X-Axis is C-Array, Y-Axis is Positive_Class_Weight'
	ml.prettyPrintGrid(C_array,class_weights,scores)
	print 'Max Score: '+str(round(max_score,2))


######## Generates sliding window position objects #############
## Look same as full length but only for the specified amount of time
def windowPositionsFromGetPositions(getPositionResult,length,windowsize):
	play = getPositionResult
	windows = generatePlayWindows(play['play_start'],play['play_end'],length,windowsize)
	new_plays = []
	for window in windows:
		start = window[0]
		end = window[1]
		new_plays.append(convertgetPositionToCustomTime(play,start,end))
	return new_plays

##### Converts a full possession into a possession that fits within the window #####
def convertgetPositionToCustomTime(play_obj,start,end):
	play = play_obj.copy()
	# 1. Filter out easy_positions
	easy_positions = play['easy_positions']
	new_easy_positions = []
	for ep in easy_positions:
		if ep['time'] >= end and ep['time'] <= start:
			new_easy_positions.append(ep)
	play['easy_positions'] = new_easy_positions
	
	# 2. Filter out events
	new_events = []
	for event in play['events']:
		if event.clock >= end and event.clock <= start:
			new_events.append(event)
	play['events'] = new_events

	# 3. Filter out Positions
	new_positions = []
	for position in new_positions:
		if position.time_on_clock >= end and position.time_on_clock <= start:
			new_positions.append(position)
	play['positions'] = new_positions
	
	#4. Remaining labels
	play['play_start'] = start
	play['play_end'] = end
	return play

# Generates tuples of time windows given a start time, end time, window length, and slider length #
def generatePlayWindows(start_time,end_time,window_length,slider_length):
	total_length = start_time - end_time
	windows = []
	last_start = start_time
	while True:
		if last_start-window_length < end_time:
			break
		else:
			windows.append((last_start,last_start-window_length))
			last_start = last_start-slider_length
	return windows

## Takes in a list and an index list, and returns the list containing only elements that have indexes found
## in the index list. This is used to separate training and test data
def retrieveIndexes(list,indexes):
	new_list = []
	for index,itm in enumerate(list):
		if index in indexes:
			new_list.append(itm)
	return new_list


#####################################
####### Vector Construction #########
#####################################

###########################################################################
######### Functions to get player positions throughout the possession #####
###########################################################################

# Gets relevant positions from a possession
# Uses the second after everyone crosses half court and runs for POSS_LENGTH
def getPositions(possession):
	POSS_LENGTH = 8
	positions = Position.objects.filter(game_id=possession.game_number,
			quarter=possession.period,
			time_on_clock__lt=possession.time_start,
			time_on_clock__gt=possession.time_end
		).order_by("time_on_clock").reverse()
	# Find average position at the end of the possession
	last_position = positions[max(1,len(positions)-10)] # Take positions the second before the end of the possession to determine direction of the play
	average_x = sum([p[0] for p in easyPosition(last_position,False)['team_positions']])/5.0
	far_side_of_court = True if average_x > 47 else False
	# Find when the play starts
	play_start = None
	for position in positions:
		over_half_court = True
		for player in easyPosition(position,False)['team_positions']:
			if (not far_side_of_court and player[0] >= 47) or (far_side_of_court and player[0]<47):
				over_half_court = False
		if over_half_court:
			play_start = position.time_on_clock-.5
			break		
	# Play end is greater of POSS_LENGTH or 1 second before end of possession
	play_end = max(play_start-POSS_LENGTH,last_position.time_on_clock)
	positions = positions.filter(time_on_clock__lt=float(play_start),time_on_clock__gt=float(play_end))
	positions = uniquePositions(positions)
	easy_positions=[]
	for pos in positions:
		easy_positions.append(easyPosition(pos,far_side_of_court))

	# Get all the associated events as well with a possession
	events = Event.objects.filter(game_id=possession.game_number,
		quarter=possession.period,
		clock__lt=possession.time_start,
		clock__gt=possession.time_end
	).order_by("clock").reverse()
	return {'easy_positions':easy_positions,'far_side_of_court':far_side_of_court,'play_start':play_start,'play_end':play_end,'positions':positions,'events':events,'id':possession.id}


### A dictionary with normalized positions (to adjust for near or far side of the court) as well as ball and time information ###
def easyPosition(position,flipped):
	team_one_positions = [(position.team_one_player_one_x,position.team_one_player_one_y),(position.team_one_player_two_x,position.team_one_player_two_y),(position.team_one_player_three_x,position.team_one_player_three_y),(position.team_one_player_four_x,position.team_one_player_four_y),(position.team_one_player_five_x,position.team_one_player_five_y)]
	team_two_positions = [(position.team_two_player_one_x,position.team_two_player_one_y),(position.team_two_player_two_x,position.team_two_player_two_y),(position.team_two_player_three_x,position.team_two_player_three_y),(position.team_two_player_four_x,position.team_two_player_four_y),(position.team_two_player_five_x,position.team_two_player_five_y)]
	ball = (position.ball_x,position.ball_y,position.ball_z)
	if position.team_one_id == 2:
		team_positions = team_one_positions
		defense_positions = team_two_positions
	else:
		team_positions = team_two_positions
		defense_positions = team_one_positions
	if flipped:
		for data in [team_positions,defense_positions]:
			for tuples in range(len(data)):
				data[tuples] = (90-data[tuples][0],50-data[tuples][1])
		ball = (90-ball[0],50-ball[1],ball[2])
	return {'team_positions':team_positions,'defense_positions':defense_positions,'ball':ball,'time':position.time_on_clock}

### Filters out any times that have more than one instance ###
def uniquePositions(position_objects):
	unique_times = []
	unique_positions = []
	for position in position_objects:
		if position.time_on_clock not in unique_times:
			unique_times.append(position.time_on_clock)
			unique_positions.append(position)
	return unique_positions


###########################################################################
######################## Build Pass Vectors ###############################
###########################################################################

def buildPassVectors(possession_obj,TIMES,printed):
	# Relevant Event IDs are as follows: 21:Dribble, 22:Pass, 23:Possession
	play_start = possession_obj['play_start']
	play_end = possession_obj['play_end']
	poss_length = play_start-play_end
	events = possession_obj['events']
	### Iterate through every event in the possession, generate a list of pass objects ###
	passes = []
	last_event = None
	for index,event in enumerate(events):
		if event.eventid == 22 and event.clock < play_start and event.clock > play_end:
			original_time = (last_event.clock if last_event else event.clock+.25)
			try:
				next_event = events[index+1]
			except:
				next_event = None
			final_time = (next_event.clock if next_event else event.clock-.25)
			original_ball_position = getPassMetrics(possession_obj,original_time)
			final_ball_position = getPassMetrics(possession_obj,final_time)
			closer_player = closerPlayer(final_ball_position,printed)
			distance = measure(original_ball_position['ball'],final_ball_position['ball'])['distance']
			up = True if original_ball_position['ball'][0]-final_ball_position['ball'][0] < 0 else False
			right = True if original_ball_position['ball'][1]-final_ball_position['ball'][1] < 0 else False
			time_into_play = play_start-event.clock
			#print 'Time: '+str(event.clock)+' Start: '+str(original_ball_position['ball'])+" End: "+str(final_ball_position['ball'])
			passes.append({'distance':distance,'start':original_ball_position['ball'],'end':final_ball_position['ball'],'time_into_play':time_into_play,'time':event.clock,'up':up,'right':right,'closer_player':closer_player})
	### Build the vectors from the pass objects ###
	times = generateTimesArray(TIMES,play_start,play_end)
	pass_array = [0,0,0,0,0,0,0]
	for index,passe in enumerate(passes):
		## Two options: just go chronologically, or you can also do it in time segments (I chose time segments) ##
		pass_type = passType(passe,printed)
		for index,time in enumerate(times):
			pass_time = passe['time'] 
			if pass_time >= time:
				pass_array[pass_type] += 1
				break
	if printed:
		print pass_array
	return pass_array

### Checks if there is a player closer to the rim and on the same side of the court ###
# Takes in a single possession instance
def closerPlayer(possession_instance,printed):
	pi = possession_instance
	team_positions = pi['team_positions']
	posting_player = None
	posting_player_distance = 10000
	for player in team_positions:
		if measure(player,pi['ball'])['distance'] < posting_player_distance:
			posting_player = player
			posting_player_distance = measure(player,pi['ball'])['distance']
	# Check if any players are closer and on the same side of the court
	closer_player = False
	cp_obj = None
	for player in team_positions:
		same_side = True if ((player[1] < 25 and posting_player[1] < 25) or (player[1] > 25 and posting_player[1] > 25)) else False
		closer = True if (measure(player,(5,25))['distance']+2 < measure(posting_player,(5,25))['distance']) else False
		if same_side and closer:
			closer_player = True
			cp_obj = player
	next_closest_player_distance = sorted([measure(posting_player,tp)['distance'] for tp in team_positions])[1]
	player_nearby = True if next_closest_player_distance < 10 else False
	# if printed:
	# 	print 'posting_player'+str(posting_player)
	# 	print 'closer_player'+str(cp_obj)
	return (closer_player or player_nearby)

def getPassMetrics(possession_obj,time):
	closest_instance = None
	delta = None
	easy_positions = possession_obj['easy_positions']
	for easy_position in easy_positions:
		if not delta or abs(easy_position['time']-time) < delta:
			delta = abs(easy_position['time']-time)
			closest_instance = easy_position
	return closest_instance

### Determine the pass type based on start and end positions ###
def passType(passe,printed):
	############# Updated Pass Types, to include 6 types of passes
	# (0) Interior, (1) Handoff, (2) Corner, (3) Entry, (4) Back Pass, (5) Other
	INTERIOR_THRESHOLD = 12 # feet from rim
	BASKET_POINT = (5.25,25)
	HANDOFF_THRESHOLD = 5 # total pass distance is less than X feet
	start_distance = measure(passe['start'],BASKET_POINT)['distance']
	end_distance = measure(passe['end'],BASKET_POINT)['distance']
	intercepts = interceptsFromTwoPoints(passe['start'],passe['end'])
	if printed:
		print passe
	# Is it an interior pass?
	if start_distance < INTERIOR_THRESHOLD and end_distance < INTERIOR_THRESHOLD:
		return 0
	# Is it a handoff?
	elif passe['distance'] < HANDOFF_THRESHOLD:
		return 1
	# Is it a corner pass?
	elif (passe['end'][1] < 8 or passe['end'][1] > 42) and passe['end'][0] < 10:
		return 2
	# Is it an entry pass?
	elif end_distance < INTERIOR_THRESHOLD+7 and intercepts['y'] > -3 and intercepts['y'] < 53 and not passe['closer_player']:
		return 3
	# Is it a backwards pass?
	elif passe['end'][0] > passe['start'][0]:
		return 4
	# Is it a cross court pass?
	elif ((passe['start'][1] > 25 and passe['end'][1] < 25) or (passe['start'][1] < 25 and passe['end'][1] > 25)) and abs(passe['start'][1]-passe['end'][1]) > 8:
		return 5
	# Must be a high entry then 
	else:
		return 6

### Returns the X and Y intercepts, given two points ###
def interceptsFromTwoPoints(point_one,point_two):
	p1x,p1y,p2x,p2y = float(point_one[0]),float(point_one[1]),float(point_two[0]),float(point_two[1])
	Y = p2y
	X = p2x
	p2x = p2x+.0001 if p2x==p1x else p2x
	m = ((p2y-p1y)/(p2x-p1x))
	b = Y - m*X
	x_intercept = -1*b/m if m else 100
	return {'x':x_intercept,'y':b}


#***********************************************************************#
#***** Takes snapshots at given times in the possession and counts *****#
#***** number of players in each region of the court *******************#
#***********************************************************************#

def buildPositionVectors(easy_positions,TIMES):
	play_start = easy_positions['play_start']
	play_end = easy_positions['play_end']
	poss_length = play_start-play_end
	times = generateTimesArray(TIMES,play_start,play_end)
	positions = [[0]*25]*len(times)
	tracker = 0
	for easy_position in easy_positions['easy_positions']:
		if times[tracker]+.1 >= easy_position['time']:
			features = generatePositionVector(easy_position)
			positions[tracker] = features
			tracker += 1
		if tracker == len(times):
			break
	vector_positions = []
	for pos in positions:
		for p in pos:
			vector_positions.append(p)
	return vector_positions

### Add DD Position Vectors TODO TODO TODO
def buildDDPositionVectors(easy_positions,TIMES):
	play_start = easy_positions['play_start']
	play_end = easy_positions['play_end']
	poss_length = play_start-play_end
	times = generateTimesArray(TIMES,play_start,play_end)
	final_vector = []
	for time in times:
		closest_instance = getPassMetrics(easy_positions,time)
		positions = [locationInAngleAndDistance(position) for position in closest_instance['team_positions']]
		vector = generateDegreeDistanceList(STANDARD_COURT_DD,positions)
		final_vector = final_vector+vector
	return final_vector

# ### Generates a 2d vector that has counts for player positions in the possession ###
def generatePositionVector(easy_position):
	#average_positions = averagePositionOverTime(easy_positions)
	number_boxes = 5
	box_width = 10
	features = [0]*((number_boxes**2))
	for objs in easy_position['team_positions']:
		x_value = min(number_boxes-1,int(round(float(objs[0])/box_width)))
		y_value = min(number_boxes-1,int(round(float(objs[1])/box_width)))
		index = ((x_value)*number_boxes)+y_value
		features[index] += 1
	return features

### Returns a list of tuples with the average positions of each player from a list of easy_positions ###
def averagePositionOverTime(easy_positions):
	count = len(easy_positions)
	sum_x = [0]*5
	sum_y = [0]*5
	for easy_position in easy_positions:
		team_positions = easy_position['team_positions']
		for player_index in range(len(team_positions)):
			sum_x[player_index] += team_positions[player_index][0]
			sum_y[player_index] += team_positions[player_index][1]
	average_positions = [(sum_x[index]/count,sum_y[index]/count) for index in range(5)]
	return average_positions

#***********************************************************************#
#**************** Number on ball side in the possession ****************#
#***********************************************************************#

def buildUniqueMeasureVectors(easy_positions,TIMES):
	play_start = easy_positions['play_start']
	play_end = easy_positions['play_end']
	poss_length = play_start-play_end
	times = generateTimesArray(TIMES,play_start,play_end)
	ballside,inside = [0]*len(times), [0]*len(times)
	tracker = 0
	for easy_position in easy_positions['easy_positions']:
		if times[tracker]+.1 >= easy_position['time']:
			ballside[tracker] = numberOfBallSidePlayers(easy_position)
			inside[tracker] = playersInside(easy_position)
			tracker += 1
		if tracker == len(times):
			break
	return ballside+inside

### Number of players on the same side as the ball
def numberOfBallSidePlayers(easy_position):
	top_side = 0
	for tp in easy_position['team_positions']:
		if tp[1] < 25:
			top_side += 1
	ball_top_side = True if easy_position['ball'][1] < 25 else False
	return top_side if ball_top_side else (5-top_side)

### Number of players within 18 feet of the basket
def playersInside(easy_position):
	players_inside = 0
	for players in easy_position['team_positions']:
		distance = locationInAngleAndDistance(players)[0]
		if distance < 18:
			players_inside += 1
	return players_inside

#***********************************************************************#
#**************** Closness Instances  **********************************#
#***********************************************************************#

### Build feature vector from the closeness algorithm using a box algorithm ###
def buildClosenessVectors(possession_obj,TIMES):
	easy_positions = possession_obj['easy_positions']
	closeness_obj = closeness(easy_positions)
	# number_boxes = 5
	# box_width = 10
	# features = [0]*(((number_boxes**2))+1)
	# for objs in closeness_obj:
	# 	midpoint = objs['object'][2]
	# 	x_value = min(number_boxes-1,int(round(float(midpoint[0])/box_width)))
	# 	y_value = min(number_boxes-1,int(round(float(midpoint[1])/box_width)))
	# 	index = (x_value*number_boxes)+y_value
	# 	if objs['on_ball']: # Use a single counter for on ball screens
	# 		features[-1] += 1
	# 	features[index] += 1

	points = [objs['object'][2] for objs in closeness_obj]
	features = generateDegreeDistanceList(STANDARD_COURT_DD,points)
	return features

########################
### Sequences Closeness Vectoors Categorizes the closeness over the time of the possession
### Types of positions: inside, outside, side or top?
def buildSequenceClosenessVectors(possession_obj,TIMES):
	easy_positions = possession_obj['easy_positions']
	closeness_obj = closeness(easy_positions)
	play_start = possession_obj['play_start']
	play_end = possession_obj['play_end']
	times = generateTimesArray(TIMES,play_start,play_end)
	vector = [0,0,0,0,0,0]*len(times)
	for objs in closeness_obj:
		midpoint = objs['object'][2]
		position = locationInAngleAndDistance(midpoint)
		inside = 1 if position[0] < 23 else 0
		# Court Degrees: 0-60:0, 60-120: 1,120-180:2
		court_degrees = (2 if position[1] < 60 else (1 if position[1] < 120  else 0))
		for idx in range(len(times)):
			time = times[idx]
			closeness_time = (objs['start']+objs['end'])/2.0
			if closeness_time >= time:
				vector[idx*6+(inside*court_degrees)] += 1
				break
	return vector

def closeness(easy_positions):
	THRESHOLD = 6
	# Get the distance array
	final_instances = []
	# Iterate through every combination of players
	for p1 in range(1,6):
		for p2 in range(p1+1,6):
				objs = findClosenssInstances(p1,p2,easy_positions,THRESHOLD)
				for obj in objs:
					final_instances.append(obj)
	return final_instances

# Finds instances where players get close together
def findClosenssInstances(player_no_1,player_no_2,easy_positions,THRESHOLD):
	# Filters all distances in the position to only those where the two players were close together
	distances = []
	for position in easy_positions:
		team_position = position['team_positions']
		dist = measure(team_position[player_no_1-1],team_position[player_no_2-1])
		if dist['distance'] < THRESHOLD:
			# Distance format: (time,min_distance,(Halfway_x,Halfway_y),(BallPos_X,BallPos_Y),player_1,player_2)
			distances.append((position['time'],dist['distance'],dist['halfway'],position['ball'],team_position[player_no_1-1],team_position[player_no_2-1]))
	
	# Group distance objects into larger instances of coming together, stored in instances
	instances = []
	temp_array = []
	for d in range(len(distances)-1):
		current = distances[d]
		next = distances[d+1]
		if current[0]-next[0] <= .12: # Threshold is 3x to avoid a player moving outside the threshold for a single frame then coming back, and counting that as two separate instances
			temp_array.append(current)
		else:
			instances.append(temp_array)
			temp_array = []
	if temp_array: # Don't forget to add the last one
		instances.append(temp_array)

	# Extract details from instances, including the min distance, length, boolean if on ball, etc. 
	objs = []
	for instance in instances:
		if instance:
			min_distance = (None,10000)
			length = instance[0][0]-instance[-1][0]
			on_ball = False
			first_instance = instance[0]
			if measure(first_instance[3],first_instance[4])['distance'] < 3 or measure(first_instance[3],first_instance[5])['distance'] < 3:
				on_ball=True
			for inst in instance:
				if inst[1] < min_distance[1]:
				 	min_distance = inst
			if length > .2:
				dict = {'start':instance[0][0],'end':instance[-1][0],'on_ball':on_ball,'object':min_distance,'length':length,'player_1':player_no_1,'player_2':player_no_2}
				objs.append(dict)
	return objs


#***********************************************************************#
#**************** Miscelleanous Helper Functions ***********************#
#***********************************************************************#


# Distance and midpoint between two positions
def measure(p1,p2):
	p1x = p1[0]
	p1y = p1[1]
	p2x = p2[0]
	p2y = p2[1]
	x = (p1x-p2x)**2
	y = (p1y-p2y)**2
	halfway = ((p1x+p2x)/2,(p1y+p2y)/2)
	return {'distance':(x+y)**.5,'halfway':halfway}

## Distance from the hoop and radians in relation to the basket
def locationInAngleAndDistance(position):
	DEGREE_MINIMIZER = 1 # Need to optimize this variable
	# Flip position if on the wrong side of the court
	player_x,player_y = position
	if player_x < 48:
		abs_pos = (player_x,player_y)
	else:
		abs_pos = (90-player_x,50-player_y)

	# Convert to radians from basket and distance
	distance = (((abs_pos[0]-0)**2)+((abs_pos[1]-25.0)**2))**0.5
	angle_x_offset = float(25-abs_pos[1])
	angle_y_offset = float(abs_pos[0])
	degrees = math.atan(angle_y_offset/angle_x_offset)*(180/math.pi) if angle_x_offset else 90
	degrees = degrees+180 if degrees < 0 else degrees
	#degrees = degrees / DEGREE_MINIMIZER
	return (distance,degrees)

### Generates time arrays from a time list
def generateTimesArray(times_array,play_start,play_end):
	times = []
	for time in times_array:
		times.append(max(play_start-time,play_end))
	return times

#***********************************************************************#
#**************** F Score **********************************************#
#***********************************************************************#
# beta = 2 weights recall higher than precision
# beta = .5 weights precision more than recall
# Precision = True Positives / (True Positives and True Negatives)
# Recall = True Positives / (True Positives and False Negatives)
def fScore(beta,true_positive,false_negative,false_positive,true_negative):
	numerator = (1+(beta**2))*true_positive
	denominator = numerator+((beta**2)*false_negative)+false_positive
	return round(float(numerator)/denominator,2)


#***********************************************************************#
#**************** Degree Distance Breakdown ****************************#
#***********************************************************************#
# court_dic = {10:[60,90,120],20:[45,135],100:[60,90,120]}
# points = [(8,80),(14,94),(25,178)]
# output is a list of counts for each of the distances, so the example is [0,1,0,0,0,1,0,0,0,0,1]
def generateDegreeDistanceList(court_dic,points):
	dic_ranges = []
	for key,value in court_dic.items():
		for degree in value:
			dic_ranges.append([key,degree,0])
		dic_ranges.append([key,360,0])
	sorted_dic_ranges = sorted(dic_ranges)
	for point in points:
		unplaced = True
		idx = 0
		while unplaced:
			if sorted_dic_ranges[idx][0] > point[0] and sorted_dic_ranges[idx][1] > point[1]:
				sorted_dic_ranges[idx][2] += 1
				unplaced = False
			else:
				idx += 1
	#print sorted_dic_ranges
	return [sdc[2] for sdc in sorted_dic_ranges]

#***********************************************************************#
#**************** Tools For Finding Events *****************************#
#***********************************************************************#

def positions(possession_id):
	possession = Possession.objects.get(id=possession_id)
	positions = getPositions(possession)
	for p in positions['easy_positions']:
		print p['time']
		print p['ball']

def events(possession_id):
	possession = Possession.objects.get(id=possession_id)
	start = possession.time_start+2
	end = possession.time_end-2
	print start
	print end
	events = Event.objects.filter(game_id=possession.game_number,
		quarter=possession.period,
		clock__lt=possession.time_start+2,
		clock__gt=possession.time_end-2
	).order_by("clock").reverse()
	for event in events:
		print event

def passevents(possession_id):
	possession = Possession.objects.get(id=possession_id)
	positions = getPositions(possession)
	print buildPassVectors(positions,[0,1,2,3,4,5,6],True)


