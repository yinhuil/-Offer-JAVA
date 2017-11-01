package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.Stack;

/*题目描述

Given a binary tree, return the postorder traversal of its nodes' values.
For example:
Given binary tree{1,#,2,3},
   1
    \
     2
    /
   3

return[3,2,1].
Note: Recursive solution is trivial, could you do it iteratively?*/
public class binaryTreePostorderTraversal {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
//	第一种方法是递归的方法，进行后续遍历
	public ArrayList<Integer> postorderTraversal(TreeNode root) {
		ArrayList<Integer> list=new ArrayList<Integer>();
		if(root==null) return list;
		help(list,root);
		return list;
	}
	private void help(ArrayList<Integer> list, TreeNode root) {
		if(root.left!=null){
			help(list, root.left);
		}
		if(root.right!=null){
			help(list,root.right);
		}
		list.add(root.val);
		
	}
//	第二种方法，非递归的方法；
	public ArrayList<Integer> postorderTraversal2(TreeNode root) {
		Stack<TreeNode> sta=new Stack<TreeNode>();
		ArrayList<Integer> list=new ArrayList<Integer>();
		TreeNode last=root;
		while(root!=null || !sta.isEmpty()){
			while(root!=null){
				sta.push(root);
				root=root.left;
			}
			root=sta.peek();
			if(root.right==null || root.right==last){
				list.add(root.val);
				sta.pop();
				last=root;
				root=null;
			}else{
				root=root.right;
			}
		}
		return list;
	}
}


















