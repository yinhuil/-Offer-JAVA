package leetcodeForJianzhi;
/*题目描述

Given a binary tree containing digits from0-9only, each root-to-leaf path could 
represent a number.An example is the root-to-leaf path1->2->3which represents the number123.
Find the total sum of all root-to-leaf numbers.
For example,
    1
   / \
  2   3

The root-to-leaf path1->2represents the number12.
The root-to-leaf path1->3represents the number13.
Return the sum = 12 + 13 =25.*/

public class sumRootToLeafNnumbers {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
//	本题使用递归的方法，前序遍历
	public static int sumNumbers(TreeNode root) {
		return helper(root,0);
	}
	private static int helper(TreeNode root, int sum) {
		if(root==null) return 0;
		sum=sum*10+root.val;
		if(root.left==null && root.right==null) return sum;
		return helper(root.left, sum)+helper(root.right, sum);
	}
	public static void main(String[] args) {
		TreeNode root1=new TreeNode(1);
		TreeNode root2=new TreeNode(2);
		TreeNode root3=new TreeNode(3);
		root1.left=root2;
		root1.right=root3;
		System.out.println(sumNumbers(root1));
	}
}
