package JianZhiOffer;

public class Serialize {
/*	题目描述
	请实现两个函数，分别用来序列化和反序列化二叉树*/	
	static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){
			val=x;
		}
	}	
	public static int index = -1;
//根据二叉树的前序遍历，因为这样可以第一个找到根节点，
	public static String Serialize(TreeNode root) {
		StringBuffer sb=new StringBuffer();
		if(root==null){
			sb.append("#,");
			return sb.toString();
		}
		sb.append(root.val+",");
		sb.append(Serialize(root.left));
		sb.append(Serialize(root.right));
		return sb.toString();
	}
	public static TreeNode Deserialize(String str){
		index++;
		int lens=str.length();
		if(index>lens){
			return null;
		}
		String[] str1=str.split(",");
		TreeNode tree=null;
		if(!str1[index].equals("#")){
			tree=new TreeNode(Integer.valueOf(str1[index]));
			tree.left=Deserialize(str);
			tree.right=Deserialize(str);
		}
		return tree;
		
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
		String str=Serialize(root1);
		TreeNode node=Deserialize(str);
		System.out.println(str);
	}
}
