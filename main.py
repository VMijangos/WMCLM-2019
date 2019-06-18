from os import listdir
import numpy as np
from nphones import nPhones
from neural_model import Model
import argparse

def parse_args():
	#Parses model arguments
	parser = argparse.ArgumentParser(description="Run model.")

	parser.add_argument('--input', type=str, nargs='?', default='corpora',
	                    help='Input corpora path')

	parser.add_argument('--output', nargs='?', default='results/results.csv',
	                    help='Output path')

	parser.add_argument('--emb_dim', type=int, default=300,
	                    help='Number of dimensions in embedding vectors. Default is 300.')

	parser.add_argument('--hid_dim', type=int, default=100,
	                    help='Number of dimensions in hidden layer. Default is 100.')

	parser.add_argument('--n', type=int, default=3,
	                    help='n-phones size. Default is 3.')

	parser.add_argument('--subsample_siz', type=int, default=100,
	                    help='Number of examples for epoch in SGD. Default is 100.')

	parser.add_argument('--lr', type=float, default=0.1,
                    	help='Learning rate for optimization. Default is 10.')

	parser.add_argument('--iter', default=50, type=int,
                      help='Number of epochs in the neural model')

	return parser.parse_args()

def main(args):
	#Path from corpora
	path = args.input 
	all_files = listdir(path)
	
	#Open output file
	f = open(args.output, 'w')
	print('file\tENT', file=f)

	for name in all_files: 
		#Name of file
		inputcorpus = path+'/'+name
		#Open the file by name
		file = open(inputcorpus,'r', encoding="utf-8")

		#extract nphones
		phones = nPhones(file.read(),nphone_siz=args.n)
		phones.get_phones()

		#Learning Bengio model
		model = Model(phones.word_phones, dim=args.emb_dim, nn_hdim=args.hid_dim)
		model.train(its=args.iter,eta=args.lr,batch=args.subsample_siz)

		#Size of nphones
		N = len(phones.voc)

		#Initilize parameters
		mu = np.zeros(N)
		Hnorm = np.zeros(N)
		H = np.zeros(N)
		for i,w in enumerate(model.voc.keys()):
		  #probabilities
		  pw = model.forward([w])
		  # \sum_j p_ij*logN p_ij
		  condHnorm = np.dot(pw,np.log(pw))/np.log(N)
		  condH = np.dot(pw,np.log(pw))
		  #Filling pre-conditional_entropies
		  Hnorm[i] = condHnorm
		  H[i] = condH
		  
		  #Filling pre-mu
		  mu += pw
		    
		#\sum_i mu_i \sum_j p_ij*logN p_ij
		Entropy = -np.dot(mu/mu.sum(0),H)
		Entropy_norm = -np.dot(mu/mu.sum(0),Hnorm)
		
		#print data
		print(name,'\t',Entropy_norm, file=f)	
		print('Corpus:',name)
		print('\tEntropy:', Entropy)
		print('\tNomralized entropy:', Entropy_norm)

	#Close output file
	f.close()

if __name__ == "__main__":
	args = parse_args()
	main(args)
