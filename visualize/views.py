from django.shortcuts import render_to_response, redirect
from django.template.loader import render_to_string
from playanalyze.models import *
from playanalyze.helper import *
from playanalyze.analyze import *
from visualize.models import *
import json
from django.template import RequestContext, Context, loader
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.utils.html import escape
from django.shortcuts import render
from django.core.exceptions import ObjectDoesNotExist


TEAM_KEY = {1:'Atlanta',
			2:'Boston',
			3:'New Orleans',
			4:'Chicago',
			5:'Cleveland',
			6:'Dallas',
			7:'Denver',
			8:'Detroit',
			9:'Golden State',
			10:'Houston', 
			11:'Indiana', 
			12:'Los Angeles Clippers', 
			13:'Los Angeles Lakers',
			14:'Miami',
			15:'Milwaukee',
			16:'Minnesota',
			17:'New Jersey',
			18:'New York',
			19:'Orlando',
			20:'Philadelphia',
			21:'Phoenix', 
			22:'Portland', 
			23:'Sacramento', 
			24:'San Antonio',
			25:'Oklahoma City',
			26:'Utah',
			27:'Washington', 
			28:'Toronto',
			29:'Memphis', 
			30:'Charlotte'}

@login_required
def visual(request):
	possessions = Possession.objects.exclude(play=None).order_by('id')
	#possessions = []
	# Get counts of each play
	counts = {}
	for poss in possessions:
		poss.play_name = poss.play.play_name
		if poss.play.play_name not in counts:
			counts[poss.play.play_name] = 1
		else:
			counts[poss.play.play_name] = counts[poss.play.play_name] + 1
	counts = sorted(counts.items(),key=lambda x:x[1],reverse=True)
	return render_to_response('index.html',{'possession_names':counts,'possessions':possessions},context_instance=RequestContext(request))

@login_required
def loadpossession(request):
	if request.method == 'GET':
		_type = request.GET.get('type','possession')
		_id = request.GET.get('id',0)
	try:
		possession = Possession.objects.get(id=_id)
	except:
		return HttpResponse(json.dumps({'status':501}), content_type='application/json')
	positions = getPositions(possession)
	data = createVisualData(possession,positions)
	return HttpResponse(json.dumps({'status':200,'play_array':data['play_array'],'closeness':closeness(positions['easy_positions']),'meta_data':data['meta_data']}), content_type='application/json')

def createVisualData(possession,getpositions):
	play_array = []
	eps = strongSidePositions(getpositions)['easy_positions']
	positions = getpositions['positions']
	pos_o = positions[0]
	home = True if positions[0].team_one_id == 2 else False
	meta_data = {'game_id':possession.game_number,
		'team_one':TEAM_KEY[int(positions[0].team_one_id)],
		'team_two':TEAM_KEY[int(positions[0].team_two_id)],
		'quarter':possession.period,
		'possession_id':possession.id,
		'possession_name':possession.play.play_name
	}
	for instance in eps:
		tp = instance['team_positions']
		dp = instance['defense_positions']
		pos_one = [1,tp[0][0],tp[0][1],
			2,tp[1][0],tp[1][1],
			3,tp[2][0],tp[2][1],
			4,tp[3][0],tp[3][1],
			5,tp[4][0],tp[4][1]]
		pos_two = [6,dp[0][0],dp[0][1],
			7,dp[1][0],dp[1][1],
			8,dp[2][0],dp[2][1],
			9,dp[3][0],dp[3][1],
			10,dp[4][0],dp[4][1]]
		combined = pos_one+pos_two
		play_array.append(combined+[instance['ball'][0],instance['ball'][1],instance['ball'][2],instance['time']])
	return {'play_array':play_array,'meta_data':meta_data}



