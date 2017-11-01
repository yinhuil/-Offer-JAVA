package JianZhiOffer;

import java.util.Arrays;

public class verifySquenceOfBST24 {
/*	输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,
	否则输出No。假设输入的数组的任意两个数字都互不相同。*/
	public static boolean verifySquenceofBst(int[] sequence, int length) {
		if(sequence==null || length<=0) return false;
		int temp=sequence[length-1];
		int i=0;
		for(;i<length-1;i++){
			if(sequence[i]>temp)
				break;
		}
		int j=i;
		for(;j<length-1;j++){
			if(sequence[j]<temp) return false;
		}
		boolean left=true;
		if(i>0){
			left=verifySquenceofBst(sequence, i);
		}
		boolean right=true;
		if(j<length-1){
			right=verifySquenceofBst(Arrays.copyOfRange(sequence, i, sequence.length-1),length-i-1);
		}
		return (left&right);
	}

//	自己喜欢第二种方法
//	题解：  关于二叉树的后序遍历，  最后一个数大于前面的一半，小于后面的一半，  当发现后面的一半出现大于的时候   就是false 反之为true；
	public static boolean VerifySquenceOfBST(int[] sequence) {
		if(sequence.length==0) return false;
		return help(sequence,0,sequence.length-1);
	}

	private static boolean help(int[] arr, int low, int high) {
		int parti=arr[high];
		int i=low;
		for(;i<high;i++){
			if(arr[i]>parti) break;
		}
		int j=i;
		for(;j<high;j++){
			if(arr[j]<parti) return false;
		}
		boolean left=true;
		if(i>0){
			left=help(arr, low, i-1);
		}
		boolean right=true;
		if(j==high-1){
			right=help(arr, i, j);
		}
		return left&&right;
		
		
	}
	public static void main(String[] args) {
		int[] ans={5,7,6,9,11,10,8};
		System.out.println(verifySquenceofBst(ans,7));
	}
}
