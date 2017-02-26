
def tree_traversal(root):
	arr = []
	terminal_map = {}
	non_terminal_map = {}
	token_list = inorder(None, root, arr, terminal_map, non_terminal_map)
	return token_list, terminal_map, non_terminal_map


def convert_to_token(parent, node, terminal_map, non_terminal_map):	
	T_i = "EMPTY"
	N_i_termChild = 0
	if node.data.get("value", None) is not None:
		try:
			v = str(node.data["value"])
		except:
			v = '<NON_ASCII>'
		T_i = v
		N_i_termChild = 1

	if T_i not in terminal_map:
		terminal_map[T_i] = len(terminal_map)

	N_i_rightSib = 0
	if parent is not None:
		if parent.right_child is not None:
			N_i_rightSib = 1

	
	N_i = (node.data["type"], N_i_termChild, N_i_rightSib)
	
	if N_i not in non_terminal_map:
		non_terminal_map[N_i] = len(non_terminal_map)

	token = (non_terminal_map[N_i], terminal_map[T_i])
	return token


def inorder(parent, node, arr, terminal_map, non_terminal_map):
	if not node: return None
	
	inorder(node, node.left_child, arr, terminal_map, non_terminal_map) #left child

	token = convert_to_token(parent, node, terminal_map, non_terminal_map)
	arr.append(token)

	inorder(node, node.right_child, arr, terminal_map, non_terminal_map) #right child
	return arr
