package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.List;

/*题目描述

Two elements of a binary search tree (BST) are swapped by mistake.
Recover the tree without changing its structure.
Note: 
A solution using O(n ) space is pretty straight forward. Could you devise a 
constant space solution?

confused what"{1,#,2,3}"means? > read more on how binary tree is serialized on OJ.

OJ's Binary Tree Serialization:
The serialization of a binary tree follows a level order traversal, where '#' signifies 
a path terminator where no node exists below.
Here's an example:
   1
  / \
 2   3
    /
   4
    \
     5
The above binary tree is serialized as"{1,2,3,#,#,4,#,#,5}".*/
public class recoverBinarySearchTree {
	static class TreeNode{
		TreeNode left;
		TreeNode right;
		int val;
		TreeNode(int x){
			val=x;
		}
	}
	public void recoverTree(TreeNode root) {
        //可以和上一题 的思路一样，还是通过中序遍历 来做
        ArrayList<TreeNode> list = new ArrayList<TreeNode>();
        if(root == null){
            return;
        }
        inorder(root,list);
        //找到两个出错的节点
        int i;
        int j;
        for ( i = 0; i < list.size() - 1; i ++) {
            if(list.get(i).val > list.get(i + 1).val) {
                break;
            }
        }
        for (j = list.size() - 1; j > 0; j --) {
            if(list.get(j).val < list.get(j - 1).val) {
                break;
            }
        }
        int temp = list.get(i).val;
        list.get(i).val = list.get(j).val;
        list.get(j).val = temp;
    }
    private static void inorder(TreeNode root,List list){
        if(root != null){
            inorder(root.left,list);
            list.add(root);
            inorder(root.right,list);
        }
    }
}


















