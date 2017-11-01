package leetcodeForJianzhi;

import java.util.ArrayList;

/*题目描述

Given a binary tree and a sum, find all root-to-leaf paths where 
each path's sum equals the given sum.
For example:
Given the below binary tree andsum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
return
[
   [5,4,11,2],
   [5,8,4,5]
]*/

//解题思路：  利用广度优先的思想进行遍历
public class pathSumII {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}

	public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
		ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> list=new ArrayList<Integer>();
		helper(root,sum,result,list);
		return result;
	}

	private void helper(TreeNode root, int sum,
			ArrayList<ArrayList<Integer>> result, ArrayList<Integer> list) {
		if(root==null) return;
		list.add(root.val);
		if(root.left==null && root.right==null && sum==root.val){
			result.add(new ArrayList<Integer>(list));
		}else{
			helper(root.left, sum-root.val, result, list);
			helper(root.right, sum-root.val, result, list);
		}
		list.remove(list.size()-1);
		
	}
}
















