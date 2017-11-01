package JianZhiOffer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class ZigzagLevelOrder {
	static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){
			val=x;
		}
	}
	public static List<Integer> ZigzagLevel(TreeNode root){
		if(root==null) return new ArrayList<Integer>();
		Queue<TreeNode> queue=new LinkedList<TreeNode>();
		queue.add(root);
		List<Integer> list=new ArrayList<Integer>();
		while(!queue.isEmpty()){
			TreeNode temp=queue.poll();
			list.add(temp.val);
			if(temp.left!=null){
			queue.add(temp.left);}
			if(temp.right!=null){
			queue.add(temp.right);}
		}
		return list;
	}
	public static List<List<Integer>> zigzagLevelOrder(TreeNode root){
		List<List<Integer>> result=new ArrayList<List<Integer>>();
		if(root==null) return result;
		Queue<TreeNode> queue=new LinkedList<TreeNode>();
		queue.add(root);
		boolean order=true;
		while(true){
			int length=queue.size();
			List<Integer> list=new ArrayList<Integer>();
			if(length==0){
				break;
			}
			while(length>0){
				TreeNode temp=queue.poll();
				if(order){
				     list.add(temp.val);}else{
				    	 list.add(0,temp.val);
				     }
				if(temp.left!=null){
					queue.add(temp.left);
				}
				if(temp.right!=null){
					queue.add(temp.right);
				}
				length--;
			}
			result.add(new ArrayList<Integer>(list));
			order=order?false:true;
		}
		return result;
	}
	public static void main(String[] args) {
		TreeNode root1=new TreeNode(1);
		TreeNode root2=new TreeNode(2);
		TreeNode root3=new TreeNode(3);
		TreeNode root4=new TreeNode(4);
		TreeNode root5=new TreeNode(5);
		TreeNode root6=new TreeNode(6);
		TreeNode root7=new TreeNode(7);
		root1.left=root2;
		root1.right=root3;
		root2.left=root4;
		root2.right=root5;
		root3.left=root6;
		root3.right=root7;
		List<List<Integer>> result=zigzagLevelOrder(root1);
		for (List<Integer> list : result) {
			System.out.println(list);
		}
	
	}
	   public List<List<Integer>> zigzagLevelOrder2(TreeNode root) {
			 List<List<Integer>> list=new ArrayList<List<Integer>>();
			 helper(root,1,list,true);
			 return list;
		   }
		private void helper(TreeNode root, int i, List<List<Integer>> list,boolean bo) {
			if(root==null) return;
			if(i>list.size()){
			  list.add(new ArrayList<Integer>());	
			}
			if(bo){
			list.get(i-1).add(root.val);
			    
			}else{
			 list.get(i-1).add(0,root.val);
			}
		
			helper(root.left, i+1, list,!bo);
			helper(root.right, i+1, list,!bo);

		}
  
}
