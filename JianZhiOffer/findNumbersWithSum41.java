package JianZhiOffer;

import java.util.ArrayList;
//将数组的第一个值和最后一个值拿出来，如果他们之间的和正好是所求的和，说明是所需要的。  如果大于所求的和，就end--，如果小于所求的和就start++；
public class findNumbersWithSum41 {
	public static void findNunberWithSum(int[] sortedArray,int number){
		if(sortedArray==null) return;
		int end=sortedArray.length-1;
		int start=0;
		while(end>start){
			int tempSum=sortedArray[start]+sortedArray[end];
			if(tempSum==number){
				System.out.println(sortedArray[start]+"******"+sortedArray[end]);
				break;
			}
			else {
				if(tempSum>number){
					end--;
				}else{
					start++;
				}
			}
			
		}
	}
//	先存储前两个数，如果两个数的和是所求的s，则说明此时的几个值时所求的， 如果小于所求的s，就small不变，然后将big往后加，这样就可以增大整体的和
//	如果大于所求的s，就将small增大，big不变，这样的话就可以使整体的数值变小。（限制条件是如果small大于s的一半时，退出循环，
//	因为如果small大于一半，再
//	增加一个数值的和肯定大于s）
	public static void findNumberWithSum2(int s){
		if(s<3) return;
		int small=1;
		int big=2;
		while(small<(s+1)/2){
			int cursum=0;
			for(int i=small;i<=big;i++){
				cursum+=i;
			}
			if(cursum==s){
				System.out.println("find one");
				for(int i=small;i<=big;i++){
					System.out.print(i+"+");
				}
				small++;
			}else{
				if(cursum>s){
					small++;
				}else{
					big++;
				}
			}
		}
	}
	public static void main(String[] args) {
		int[] in={1,2,4,7,11,15};
		findNunberWithSum(in, 15);
		findNumberWithSum2(15);
		ArrayList<Integer> list=new ArrayList<Integer>();
		
	}
	
}
