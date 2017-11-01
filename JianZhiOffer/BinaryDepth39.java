package JianZhiOffer;

public class BinaryDepth39 {
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
//	采用递归的方法遍历左子树和右子树，判断左右的大小，最后找到最深的长度
	public static int binaryTreeDepth(TreeNode root){
		if(root==null)
			return 0;
		int left=binaryTreeDepth(root.left);
		int right=binaryTreeDepth(root.right);
		return (left>right)?(left+1):(right+1);
	}
	public static boolean isBalance(TreeNode root){
		if(root==null) return true;
		int left=binaryTreeDepth(root.left);
		int right=binaryTreeDepth(root.right);
		int diff=left-right;
		if(diff>1 ||diff<-1){
			return false;
		}
		return isBalance(root.left)&& isBalance(root.right);
	}
//	采用后序遍历，检测平衡树(如果某二叉树中任意结点的左右子树的深度相差不超过1 ，那么它就是一棵平衡二叉树。)
//	public static boolean isBalancedHelper(TreeNode root,int[] depth){
//		if(root==null){
//			depth[0]=0;
//			return true;			
//		}
//		int[] left=new int[1];
//		int[] right=new int[1];
//		if(isBalancedHelper(root.left, left)&&isBalancedHelper(root.right, right)){
//			int diff=left[0]-right[0];
//			if(diff<=1 || diff>=-1){
//				depth[0]=1+(left[0]>right[0]?left[0]:right[0]);
//				return true;
//			}
//			
//		}
//		return false;
//	}
//	public static boolean isBalanced(TreeNode root){
//		int[] depth=new int[1];
//		return isBalancedHelper(root, depth);
//	}
	public static boolean isBalanced(TreeNode root) {
        int[] depth = new int[1];
        return isBalancedHelper(root, depth);
    }

    public static boolean isBalancedHelper(TreeNode root, int[] depth) {
        if (root == null) {
            depth[0] = 0;
            return true;
        }

        int[] left = new int[1];
        int[] right = new int[1];

        if (isBalancedHelper(root.left, left) && isBalancedHelper(root.right, right)) {
            int diff = left[0] - right[0];
            if (diff >= -1 && diff <= 1) {
                depth[0] = 1 + (left[0] > right[0] ? left[0] : right[0]);
                return true;
            }
        }

        return false;
    }

	public static void main(String[] args) {
		    TreeNode n1 = new TreeNode(1);
		    TreeNode n2 = new TreeNode(1);
		    TreeNode n3 = new TreeNode(1);
		    TreeNode n4 = new TreeNode(1);
		    TreeNode n5 = new TreeNode(1);
		    TreeNode n6 = new TreeNode(1);
		    TreeNode n7 = new TreeNode(1);

	        n1.right = n2;
	        n2.right = n3;
	        n3.right = n4;
	        n4.right = n5;

	        System.out.println(isBalanced(n1));
	}
}
