package JianZhiOffer;

public class FindNumsAppearOnce {
/*	题目描述
	一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。*/
	
	
//	解题思路：当找出一个数组中只出现一个数字出现了一次，其余的都出现两次的情况，我们可以用异或找到这个值
//	在本题中出现两个数字都是出现了一次，我们就可以将整体异或出来，找到最高位为1的情况下并记录这个index（因为异或得到的值，肯定是
//	一个有一个没有的情况、）
	public void FindNumsAppearOnce(int [] array,int num1 , int num2) {
		if(array==null || array.length<2){
			return;
		}
		int temp=0;
		for(int i=0;i<array.length-1;i++){
			temp^=array[i];
		}
		int indexOf1=findFirstBite(temp);
		for(int i=0;i<array.length-1;i++){
			if(isBit(array[i],indexOf1)){
				num1^=array[i];
			}else{
				num2^=array[i];
			}
		}
	}	

	private boolean isBit(int num, int indexOf1) {
		num=num>>indexOf1;
		return (num&1)==1;
	}

	private int findFirstBite(int num) {
		int indexBit=0;
		while(((num & 1)==0) && (indexBit)<8*4){
			num=num>>1;
			++indexBit;
		}
		return indexBit;
	}

}
