package leetcodeForJianzhi;
/*
题目描述

Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that 
adding up all the values along the path equals the given sum.
For example:
Given the below binary tree andsum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
return true, as there exist a root-to-leaf path5->4->11->2which sum is 22.*/

//解题思路：本题采用递归的方式，首先将sum减去该节点，然后接着往下遍历，直到找到
//root.left==null && root.right==null && sum==0的情况，
public class pathSum {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
	public boolean hasPathSum(TreeNode root, int sum) {
		if(root==null) return false;
		return helper(root,sum);
	}
	private boolean helper(TreeNode root, int sum) {
		sum-=root.val;
		if(root.left==null && root.right==null && sum==0){
			return true;
		}
		boolean left=false,right=false;
		if(root.left!=null){
			left=helper(root.left, sum);
		}
		if(root.right!=null){
			right=helper(root.right, sum);
		}
		return left||right;
	}
}



























