from collections import namedtuple
import random
Experience = namedtuple('Experience',
						('states','actions','next_states','rewards','dones','history_info','cur_hidden','next_hidden', 'rnn_hidden'))

class ReplayMemory:
	def __init__(self,capacity, history_length):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.history_seq_length = history_length
		self.sampling_indexes = None
		
	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self, batch_size):
		# print(len(self.memory),batch_size)
		return random.sample(self.memory, batch_size)

	def sample_by_index(self, indexes):
		self.sampling_indexes = indexes
		return [self.memory[i] for i in self.sampling_indexes]
	
	def __len__(self):
		return len(self.memory)