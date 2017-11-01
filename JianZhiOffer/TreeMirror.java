package JianZhiOffer;

public class TreeMirror {
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
	public static void TreeMirror(TreeNode root){
		if(root==null){
			return;
		}
		if(root.left==null && root.right==null){
			return;
		}
		TreeNode temp=root.left;
		root.left=root.right;
		root.right=temp;
		if(root.left!=null){
			TreeMirror(root.left);
		}
		if(root.right!=null){
			TreeMirror(root.right);
		}
	}
	public static void printTree(TreeNode root){
		if(root==null){
			return;
		}
		System.out.print(root.val);
		printTree(root.left);
		printTree(root.right);
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
		TreeMirror(n1);
		printTree(n1);
		
		
	}
}
