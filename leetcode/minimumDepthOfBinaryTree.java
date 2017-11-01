package leetcodeForJianzhi;
/*题目描述
Given a binary tree, find its minimum depth.The minimum depth is 
the number of nodes along the shortest path from the root node 
down to the nearest leaf node*/
public class minimumDepthOfBinaryTree {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
//	解析：因为你每一次一定要照顾一棵树，如果左边或者右边没有的话，就会出现总长度为1的情况，因为他会把左边算成是最小的，其实左边没有树；
	public int run(TreeNode root) {
		if(root==null) return 0;
		int left=run(root.left);
		int right=run(root.right);
		if(left==0 || right==0)return left+right+1;
		return Math.min(left,right)+1;
	}
}
