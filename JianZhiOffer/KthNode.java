package JianZhiOffer;

import java.util.Stack;

public class KthNode {
	/*	题目描述
	给定一颗二叉搜索树，请找出其中的第k大的结点。例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。*/
	static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){
			val=x;
		}
	}
//非递归的方法;利用中序遍历的非递归形式获得
	public static TreeNode  KthNode1(TreeNode root,int k){
		Stack<TreeNode> sta=new Stack<TreeNode>();
		TreeNode temp=null;
		while(!sta.isEmpty() || root!=null){
			while(root!=null){
				sta.add(root);
				root=root.left;
			}
			 root=sta.pop();
			if(k==1){
				temp=root;
				break;
			}else{
				k--;
			}
			root=root.right;
		}
		return temp;
	}
	static int index=0;
	public static TreeNode KthNode(TreeNode pRoot, int k){
        if(pRoot!=null){
        	TreeNode node=KthNode(pRoot.left, k);
        	if(node!=null){
        		return node;
        	}
        	index++;
        	if(index==k){
        		return pRoot;
        	}
        	node=KthNode(pRoot.right, k);
        	if(node!=null){
        		return node;
        	}
        }
        return null;
    }
	
	
//剑指offer 上面的方法
	public static TreeNode Kth(TreeNode pRoot,int k){
		if(pRoot==null || k==0){
			return null;
		}
		return help(pRoot,k);
	}
	private static TreeNode help(TreeNode pRoot, int k) {
		TreeNode temp=null;
		if(pRoot.left!=null){
			temp=help(pRoot.left, k);
		}
		if(temp==null){
			if(k==1) temp=pRoot;
			k--;
		}
		if(temp==null && pRoot.right!=null){
			temp=help(pRoot.right, k);
		}
		return temp;
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
		TreeNode root=Kth(root1, 2);
		System.out.println(root.val);
		
	}

}
