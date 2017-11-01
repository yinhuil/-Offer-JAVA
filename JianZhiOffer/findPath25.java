package JianZhiOffer;

import java.util.ArrayList;
import java.util.List;
//本题是深度优先，对于深度优先就是使用以下的方法，进行递归。     而对于广度优先则使用queue的先进先出的原则
public class findPath25 {
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
	public static List<List<Integer>> findPath(TreeNode root,int sum){
	   	List<List<Integer>> result=new ArrayList<List<Integer>>();
	   	List<Integer> list=new ArrayList<Integer>();
	   	findPathHelper(root,sum,result,list);
	   	return result;
	}
	public static void findPathHelper(TreeNode root, int sum,
			List<List<Integer>> result, List<Integer> list) {
		if(root==null)
			return;
		list.add(root.val);
		if(root.left==null && root.right==null && sum==root.val){
			result.add(new ArrayList<Integer>(list));
		}else{
			findPathHelper(root.left, sum-root.val, result, list);
			findPathHelper(root.right, sum-root.val, result, list);
		}
		list.remove(list.size()-1);
	}
	
	
//	第二种方法，跟第一种一样，但是简略点；

	private static ArrayList<ArrayList<Integer>> listAll=new ArrayList<ArrayList<Integer>>();
	private static ArrayList<Integer> list=new ArrayList<Integer>();
	public static ArrayList<ArrayList<Integer>> findpath1(TreeNode root,int target){
		if(root==null) return listAll;
		list.add(root.val);
		target-=root.val;
		if(target==0 && root.left==null && root.right==null){
			listAll.add(new ArrayList<Integer>(list));
		}
		findpath1(root.left, target);
		findpath1(root.right,target);
		list.remove(list.size()-1);
		return listAll;
	}
	
	public static void main(String[] args) {
		TreeNode n1=new TreeNode(10);
		TreeNode n2=new TreeNode(5);
		TreeNode n3=new TreeNode(12);
		TreeNode n4=new TreeNode(4);
		TreeNode n5=new TreeNode(7);
		n1.left=n2;
		n1.right=n3;
		n2.left=n4;
		n2.right=n5;
		ArrayList<ArrayList<Integer>> result=findpath1(n1, 22);
		for (ArrayList<Integer> list : result) {
			System.out.println(list.toString());
		}
	}

}
