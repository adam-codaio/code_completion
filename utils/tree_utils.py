import json

class LCRSNode():

	def __init__(self, node_id, data):
		self.id = node_id
		self.data = data
		self.left_child = None
		self.right_child = None

	def add_left(self, node):
		self.left_child = node
		return node

	def add_right(self, node):
		self.right_child = node
		return node

	def __repr__(self, level=0):
		ret = "\t"*level+str(self.id)+"\n"
		if self.left_child is not None:
			ret += self.left_child.__repr__(level+1)
		if self.right_child is not None:
			ret += self.right_child.__repr__(level+1)
		return ret

class LCRSTree():

	def __init__(self, root):
		self.root = root

def constructTree(ast):
	children_dic = dict()
	nodes = dict()
	for data in ast:
		if data == 0:
			continue
		node_id = data.pop('id', None)
		children = data.pop('children', [])
		tree_node = LCRSNode(node_id, data)
		children_dic[node_id] = children
		nodes[node_id] = tree_node

	tree = LCRSTree(nodes[0])
	for node_id, children in children_dic.iteritems():
		if len(children) > 0:
			curr = nodes[node_id].add_left(nodes[children[0]])
			if len(children) > 1:
				for child in children[1:]:
					curr = curr.add_right(nodes[child])

	return tree

def ast_to_lcrs(ast):
	tree = constructTree(ast)
	return tree

if __name__ == "__main__":
	main()