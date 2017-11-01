package JianZhiOffer;

public class HasSubtree18 {
	/*输入两棵二叉树A，B，判断B是不是A的子结构。
	（ps：我们约定空树不是任意一个树的子结构）*/	
	static class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}
//  本道题的方法是，递归往下进行 很简单；
	public static boolean HasSubtree(TreeNode root1, TreeNode root2) {
		boolean result=false;
		if(root1 !=null && root2!=null){
			if(root1.val==root2.val){
				result=DoesTreeHaveTree(root1,root2);
			}
			if(!result){
				result=HasSubtree(root1.left, root2);
			}
			if(!result){
				result=HasSubtree(root1.right, root2);
			}
		}
		return result;
	}

	private static boolean DoesTreeHaveTree(TreeNode root1, TreeNode root2) {
		if(root2==null) return true;
		if(root1==null) return false;
		if(root1.val!=root2.val){
			return false;
		}
		return DoesTreeHaveTree(root1.left, root2.left) && DoesTreeHaveTree(root1.right, root2.right);
	}
}
