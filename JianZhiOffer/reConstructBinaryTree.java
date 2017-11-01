package JianZhiOffer;

public class reConstructBinaryTree {
	/*输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
	假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}
	和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。*/
	static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x){
			val=x;
		}
	}
//解析，前序遍历来说，第一个是根节点并确定值时temp， 在中序遍历当中，找到属于这个值temp1=temp，那么temp1的左边就是左子树，右边就是右子树，
//	在根据左子树的长度，在前序遍历数组中确定左子树，剩下的就是右子树了
   public static TreeNode reConstructBinaryTree(int[] pre,int[] in){
	   TreeNode root=reconstruct(pre,0,pre.length-1,in,0,in.length-1);
	   return root;
   }

	private static TreeNode reconstruct(int[] pre, int startPre, int endPre, int[] in,
			int startIn, int endIn) {
		if(startPre>endPre || startIn>endIn){
			return null;
		}
		TreeNode root=new TreeNode(pre[startPre]);
		for(int i=startIn;i<=endIn;i++){
			if(in[i]==pre[startPre]){
				root.left=reconstruct(pre, startPre+1, startPre+i-startIn, in, startIn, i-1);
				root.right=reconstruct(pre, i-startIn+startPre+1, endPre, in, i+1, endIn);
			}
		}
		return root;
	}

}
