from hmmlearn.hmm import GMMHMM
import os
from os import listdir
from os import walk
from os.path import isfile, join
import numpy

def get_files_list(root):
        files_list = []
        for dirpath, dirnames, files in walk(root):
                for name in files:
                        if name.lower().endswith(".txt"):
                                files_list.append(join(dirpath,name))
        return files_list   
        
		
def main():
	outdir = r'./training_files/multi'
	outdir2 = r'./training_files/arnab'
	outdir3 = r'./training_files/kejriwal'
	outdir4 = r'./training_files/ravish'
	outdir5 = r'./training_files/not-shouting'
	outdir6 = r'./training_files/shouting'
	outdir7 = r'./training_files/single'
	outdir8 = r'./training_files/modi'
	outdir9 = r'./training_files/ond_more'
	
	
	#create 3 hmm one for each case
	
	multi=GMMHMM(5,2);
	discuss=GMMHMM(5,2);
	arnab=GMMHMM(5,2);
	kejriwal=GMMHMM(5,2);
	ravish=GMMHMM(5,2);
	
	
	
	notshouting=GMMHMM(5,2);
	shouting=GMMHMM(5,2);
	single=GMMHMM(5,2);
	
	#training for multi
	
	l=get_files_list(outdir)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
	
	
	multi.fit(obs)
	
	

	#training for arnab
	
	l=get_files_list(outdir2)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
	
	arnab.fit(obs)
	
	#training for kejriwal
	
	l=get_files_list(outdir3)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
	
	kejriwal.fit(obs)
	
	#training for ravish
	
	l=get_files_list(outdir4)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
			
	
	ravish.fit(obs)
	
	#training for notshouting
	
	l=get_files_list(outdir5)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
			
	
	notshouting.fit(obs)
	
	#training for shouting
	
	l=get_files_list(outdir6)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
			
	
	shouting.fit(obs)
	
	#training for single
	
	l=get_files_list(outdir7)
	
	for i in l:
		f=open(i,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
			
	
	single.fit(obs)
	
	
	#Its time for some testing
	q=[]
	t="testcase_output.txt"
	out=open(t,"w")
		
	#Read test file and make list of list of sequence 10   for --->1
	#te=["test1.txt","test2.txt","test3.txt","test4.txt","test5.txt","test6.txt","test7.txt","test8.txt","test9.txt","test10.txt"]
	
	#f=open("expected.txt")
	#d_expected={}
	'''
	
	for line in f:
		x=line.strip().split()
		d_expected[x[0]]={'arnab':float(x[1]),'kejriwal':float(x[2]),'ravish':float(x[3])}
	'''
		
	te=get_files_list(r'./testing_files')		
	#te=["test1.txt","test2.txt","test3.txt"]
	for ad in te:
		d={"arnab":0,"kejriwal":0,"ravish":0}
		f=open(ad,"r")
		obs=[]
		i_sequence=[]
		count=0
		for line in f:
			individual_obs=line.strip().split(",")
			#print individual_obs
			individual_obs=[float(i) for i in individual_obs]
			i_sequence.append(individual_obs)
			count+=1
			if count==10:
				obs.append(numpy.array(i_sequence))
				count=0
				i_sequence=[]
			
		p=[]
		p_choosen=[]
		p1_choosen=[]
		p1=[]
		p2=[]
		p2_choosen=[]
		
		#print obs
		for i in obs:
			p.append((shouting.score(i),"shouting"))
			p.append((notshouting.score(i),"notshouting"))
			p_choosen.append(max(p,key=lambda x: x[0]))
			p=[]
		for i in obs:
			p1.append((arnab.score(i),"arnab"))
			p1.append((kejriwal.score(i),"kejriwal"))
			p1.append((ravish.score(i),"ravish"))
			p1_choosen.append(max(p1,key=lambda x: x[0]))
			p1=[]	
		
		for i in obs:
			p2.append((multi.score(i),"multi"))
			p2.append((single.score(i),"single"))
			p2_choosen.append(max(p2,key=lambda x: x[0]))
			p2=[]		
		#print p
		
		
		p=[]
		p1=[]
		p_choosen=[b for a,b in p_choosen]
		p1_choosen=[b for a,b in p1_choosen]
		p2_choosen=[b for a,b in p2_choosen]
		
		'''
		#print p_choosen
		#print the state sequence with the timestamp in the output file
	
		t="testcase_output_9.txt"
		out=open(t,"a+")

		out.write(str(ad)+"--->")
		out.write(p_choosen[0])
		out.write("\n")
		'''
			
		#calculate the amount per second and append to the same file
		
		#print p_choosen
		#print p1_choosen
		shouting1=[]
		notshouting1=[]
		
		totaltime=len(p_choosen)*0.05
		
		single_count=0
		
		
		for i in range(len(p_choosen)):
			if p2_choosen[i]=="single":
				single_count+=1
				if p_choosen[i]=="shouting":
					shouting1.append(p1_choosen[i])
				elif p_choosen[i]=="notshouting":
					notshouting1.append(p1_choosen[i])
		#print d
		d_shouting={"arnab":0,"kejriwal":0,"ravish":0}
		d_notshouting={"arnab":0,"kejriwal":0,"ravish":0}
		
		for i in shouting1:
			d_shouting[i]+=1
		
		for i in notshouting1:
			d_notshouting[i]+=1
		
		#print p_choosen

		out.write("\n*******--> "+str(ad)+"  <--*******\n")
		#write arnab,ravish and kejri
		fn=ad.strip().split("/")
		fn=fn[len(fn)-1]
		
		#out.write("Time predicted for questioning: "+str((d5['question'])*0.05)+" seconds.\n")
		#out.write("Time predicted for discussion: "+str((d5['discuss'])*0.05)+" seconds.\n")
		out.write("\nChecking single HMM and multi HMM:\n")
		out.write("Number of instance of Single: "+str(single_count)+"\n")
		out.write("\nChecking shouting and non-shouting HMM for all Single instances:\n")
		out.write("Number of instance of Shouting: "+str(len(shouting1))+"\n")
		out.write("Number of instance of Not-shouting: "+str(len(notshouting1))+"\n")
		out.write("\nChecking the frequency of each speaker in both both shouting and not shouting instance...\n")
		out.write("Shouting instance: \n"+str(d_shouting)+"\n")
		out.write("Not-Shouting instance: \n"+str(d_notshouting)+"\n")
		
		out.write("\nResult:\n")
		
		for c,d in d_shouting.items():
			out.write(str(c)+" was shouting for "+str(d*0.05)+" sec.\n")
		
		out.write("\n")
		
		for c,d in d_notshouting.items():
			out.write(str(c)+" was not shouting for "+str(d*0.05)+" sec.\n")
		
		
		out.write("\n")
		for c,d in d_shouting.items():
			out.write(str(c)+" was shouting for "+str(((d*0.05)/totaltime)*100)+" % of time.\n")
		
		out.write("\n")
		
		for c,d in d_notshouting.items():
			out.write(str(c)+" was not shouting for "+str(((d*0.05)/totaltime)*100)+" sec.\n")
		
		out.write("\n")
		
		print d_shouting
		print d_notshouting
	
main()

