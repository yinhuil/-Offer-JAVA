package JianZhiOffer;


public class isSymmetrical {
/*	题目描述
	请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的*/
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
//	解析； 本题采用递归的方式，每一层的遍历
    boolean isSymmetrical(TreeNode pRoot){
        if(pRoot==null) return true;
        return help(pRoot.left,pRoot.right);
    }
	private boolean help(TreeNode left, TreeNode right) {
		if(left==null) return right==null;
		if(right==null) return false;
		if(left.val!=right.val) return false;
		return help(left.left, right.right) && help(left.right, right.left);
		
	}
	
}
