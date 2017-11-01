package JianZhiOffer;

import java.util.Stack;

/*题目描述
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。
要求不能创建任何新的结点，只能调整树中结点指针的指向。*/
public class covertLinked27 {
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
//采用中序遍历的非递归的形式。  首先找到第一个节点，然后把这个值交给pRootOfTree，last表示移动的指针，持续的往前移动，
// 然后分别将last.right=root,root.left=last实现双向链表的结构。
	public TreeNode Convert(TreeNode pRootOfTree) {
		if (pRootOfTree == null)
			return null;
		TreeNode last = null;
		TreeNode root = pRootOfTree;
		Stack<TreeNode> sta = new Stack<TreeNode>();
		boolean isfirst = true;
		while (!sta.isEmpty() || root != null) {
			while (root != null) {
				sta.push(root);
				root = root.left;
			}
			root = sta.pop();
			if (isfirst) {
				pRootOfTree = root;
				last = root;
				isfirst = false;
			} else {
				last.right = root;
				root.left = last;
				last = root;
			}
			root = root.right;
		}
		return pRootOfTree;
	}

	
	
//	采用树的中序遍历，在遍历的同时，将树的左右枝子进行连接；
	public static TreeNode baseconvert(TreeNode root, TreeNode lastNode) {
		if (root == null)
			return lastNode;
		TreeNode current = root;
		if (current.left != null) {
			lastNode=baseconvert(current.left, lastNode);
		}
		current.left = lastNode;
		if (lastNode != null)
			lastNode.right = current;
		lastNode = current;
		if (current.right != null) {
			lastNode=baseconvert(current.right, lastNode);
		}
		return lastNode;
	}
    
	public static TreeNode convert(TreeNode root) {
		TreeNode lastNode = null;
		lastNode = baseconvert(root, lastNode);
		TreeNode headNode = lastNode;
		while (headNode.left != null) {
			headNode = headNode.left;
		}
		return headNode;
	}

	public static void main(String[] args) {

		// TODO Auto-generated method stub

		TreeNode root = new TreeNode(10);

		TreeNode six = new TreeNode(6);

		TreeNode four = new TreeNode(4);

		TreeNode eight = new TreeNode(8);

		TreeNode fourteen = new TreeNode(14);

		TreeNode twelve = new TreeNode(12);

		TreeNode sixteen = new TreeNode(16);

		root.left = six;

		root.right = fourteen;

		six.left = four;

		six.right = eight;

		four.left = null;

		four.right = null;

		eight.left = null;

		eight.right = null;

		fourteen.left = twelve;

		fourteen.right = sixteen;

		twelve.left = null;

		twelve.right = null;

		sixteen.right = null;

		sixteen.left = null;

		TreeNode result = convert(root);

		// BinaryTreeNode result=baseconvert(root, null);

		while (result != null) {

			System.out.println(result.val);

			result = result.right;

		}
	}
}
