package JianZhiOffer;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class PrintTreeFromTopToBottom {

	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
	public static List<Integer> printTree(TreeNode root){
		if(root==null) return new ArrayList<Integer>();
		List<Integer> list=new ArrayList<Integer>();
		Queue<TreeNode> queue=new LinkedList<TreeNode>();
		queue.add(root);
		while(!queue.isEmpty()){
			TreeNode temp=queue.poll();
			list.add(temp.val);
			if(temp.left!=null){
				queue.add(temp.left);
			}
			if(temp.right!=null){
				queue.add(temp.right);
			}
		}
		return list;
	}
	public static void main(String[] args) {
		TreeNode n1=new TreeNode(8);
		TreeNode n2=new TreeNode(6);
		TreeNode n3=new TreeNode(10);
		TreeNode n4=new TreeNode(5);
		TreeNode n5=new TreeNode(7);
		TreeNode n6=new TreeNode(9);
		TreeNode n7=new TreeNode(11);
        n1.left=n2;
        n1.right=n3;
        n2.left=n4;
        n2.right=n5;
        n3.left=n6;
        n3.right=n7;
        
        List<Integer> result=printTree(n1);
        for (Integer integer : result) {
			System.out.println(integer);
		}
        
		

	}

}
