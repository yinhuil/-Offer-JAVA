package leetcodeForJianzhi;

import JianZhiOffer.reConstructBinaryTree.TreeNode;

/*题目描述

Given inorder and postorder traversal of a tree, construct the binary tree.
Note: 
You may assume that duplicates do not exist in the tree.*/

//解析：这里面是给了中序和后序遍历的数组，
public class buildTree {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}

	public TreeNode buildTree(int[] inorder, int[] postorder) {
		TreeNode root = reconstruct(postorder, 0, postorder.length - 1,
				inorder, 0, inorder.length - 1);
		return root;
	}

	private TreeNode reconstruct(int[] post, int postIn, int postEnd, int[] in,
			int inIn, int inEnd ) {
		if(postIn>postEnd || inIn>inEnd){
			return null;
		}
		TreeNode root=new TreeNode(post[postEnd]);
		for(int i=inIn;i<=inEnd;i++){
			if(in[i]==post[postEnd]){
				root.left=reconstruct(post, postIn, postIn+i-inIn-1, in, inIn, i-1);
				root.right=reconstruct(post, postIn+i-inIn, postEnd-1, in, i+1, inEnd);
			}
		}
		return root;
	}
	

}
