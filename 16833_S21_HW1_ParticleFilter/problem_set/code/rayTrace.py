import math
#code heavily based on work from lodev.org/cgtutor/raycasting.html
#note: assumes map pixels are 10 cm
#estimates distance to nearest surface by following traj of x until hitting surface in map
#pos=[x,y,theta] of the robot,lAngle=angle offset of laser
#also adjusts for the ~25cm offset from the actual robot position
def adjuster(reading,adj,angle):
	return math.sqrt(reading**2 + adj**2-2*reading*adj*math.cos(math.pi-angle))
def rayTrace(rayInfo):
	pos=rayInfo[0]
	lAngle=rayInfo[1]
	oMap=rayInfo[2]
	certainty=0.35
	#find current block in map
	adj=25
	laserTheta=pos[2]+lAngle
	rayDirX=math.cos(laserTheta)
	rayDirY=math.sin(laserTheta)
	infty=float("inf")
	deltaDist=[infty,infty]
	eps=0.0001
	res=10
	if abs(rayDirX)>eps:
		deltaDist[0]=abs(1/rayDirX)
	if abs(rayDirY)>eps:
		deltaDist[1]=abs(1/rayDirY)
	#initialize rayPos, position of ray in xy
	rayPos=[pos[0]/res,pos[1]/res,laserTheta]
	#position in map coordinates
	mapPos=[int(rayPos[0]),int(rayPos[1]),laserTheta]
	#length of ray to next x or y side
	sideDistX=0
	sideDistY=0
	#ray direction
	stepX=1
	stepY=1
	side=0
	if rayDirX<0:
		stepX=-1
		sideDistX=(rayPos[0]-mapPos[0])*deltaDist[0];
	else:
		stepX=1
		sideDistX=(mapPos[0]-rayPos[0]+1)*deltaDist[0];
	if rayDirY<0:
		stepY=-1
		sideDistY=(rayPos[1]-mapPos[1])*deltaDist[1];
	else:
		stepY=1
		sideDistY=(mapPos[1]-rayPos[1]+1)*deltaDist[1];
	maxX=oMap.shape[0]
	maxY=oMap.shape[1]
	withinX = mapPos[0] >= 0 and mapPos[0] < maxX
	withinY = mapPos[1]>=0 and mapPos[1] < maxY
	notWall = (withinX and withinY and oMap[mapPos[1]][mapPos[0]] < certainty)
	while(withinX and withinY and notWall):
		#jump to next map square or in x-dir or in y-dir
		if sideDistX<sideDistY:
			sideDistX+=deltaDist[0]
			mapPos[0]+=stepX
			pSide=side
			side=0
		else:
			sideDistY+=deltaDist[1]
			mapPos[1]+=stepY
			pside=side
			side=1
		withinX = mapPos[0] >= 0 and mapPos[0] < maxX
		withinY = mapPos[1]>=0 and mapPos[1] < maxY
		notWall = (withinX and withinY and oMap[mapPos[1]][mapPos[0]] < certainty)
	#can find length of ray by determining the last valid sideDistX/Y
	perpWallDist=0
	if side==0:
		perpWallDist=(mapPos[0]-rayPos[0] + (1-stepX)/2)/rayDirX
	else:
		perpWallDist=(mapPos[1]-rayPos[1] + (1-stepY)/2)/rayDirY
	#return ray length, with 25cm adjustment for laser
	finalLen=adjuster(perpWallDist*10,adj,abs(lAngle))
	return finalLen
#
#debugging code, ignore
#import matplotlib.pyplot as plt
#m=[\
#[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],\
#[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
#[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
#[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],\
#[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
#l=range(360)
#d=[0]*360
#l=[15]
#d=[0]
#for i in range(len(l)):
#	lAng=l[i]*2*math.pi/360
#	p=[105,15,0]
#	certain=0.2
#	rRes=rayTrace([p,lAng,m,certain])
#	d[i]=rRes
#	print("degree=",i," at rads ",lAng,"distance is ",rRes)
#plt.scatter(l,d)
#plt.show()
