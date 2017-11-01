package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;


/*题目描述

Given a binary tree, return the bottom-up level order traversal of 
		its nodes' values. (ie, from left to right, level by level from leaf to root).
For example:
Given binary tree{3,9,20,#,#,15,7},
    3
   / \
  9  20
    /  \
   15   7

return its bottom-up level order traversal as:
[
  [15,7]
  [9,20],
  [3],
]*/
public class binaryTreeLevelOrderTraversalII {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
//	解题思路，采用广度优先的思想，就是中间使用Queue队列来实现，
//	但是要注意保存的时候时候list带的list.add(0,目标)这样可以实现反转的效果
	public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
		ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
		if(root==null) return result;
		Queue<TreeNode> queue=new LinkedList<TreeNode>();
		queue.add(root);
		while(!queue.isEmpty()){
			ArrayList<Integer> list=new ArrayList<Integer>();
			int lens=queue.size();
			while(lens>0){
				TreeNode temp=queue.poll();
				list.add(temp.val);
				if(temp.left!=null){
					queue.add(temp.left);
				}
				if(temp.right!=null){
					queue.add(temp.right);
				}
				lens--;
			}
			result.add(0,list);
		}
		return result;
	}
}















