package leetcodeForJianzhi;
/*题目描述

Given a binary tree, find the maximum path sum.
The path may start and end at any node in the tree.
For example:
Given the below binary tree,
       1
      / \
     2   3

Return6.*/
public class binaryTreeMaximumPathSum {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}

	static int maxValue;

	public static int maxPathSum(TreeNode root) {
		maxValue = Integer.MIN_VALUE;
		maxPathDown(root);
		return maxValue;
	}

	private static int maxPathDown(TreeNode node) {
		if (node == null)
			return 0;
		int left = Math.max(0, maxPathDown(node.left));
		int right = Math.max(0, maxPathDown(node.right));
		maxValue = Math.max(maxValue, left + right + node.val);
		return Math.max(left, right) + node.val;
	}
//	下面是求一个最大路劲的程序
	public static int maxPath(TreeNode root) {
		if(root==null) return 0;
		int left=maxPath(root.left);
		int right=maxPath(root.right);
		return Math.max(left, right)+root.val;
	}
	public static void main(String[] args) {
		TreeNode root=new TreeNode(1);
		TreeNode root1=new TreeNode(2);
		TreeNode root2=new TreeNode(3);
		TreeNode root3=new TreeNode(4);
		TreeNode root4=new TreeNode(5);
		TreeNode root5=new TreeNode(6);
		TreeNode root6=new TreeNode(7);
		
		root.left=root1;
		root.right=root2;
		root1.left=root3;
		root1.right=root4;
		root2.left=root5;
		root2.right=root6;
		System.out.println(maxPathSum(root));
	}
}
