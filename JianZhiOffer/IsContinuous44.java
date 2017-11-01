package JianZhiOffer;

import java.util.Arrays;

public class IsContinuous44 {
	
/*	题目描述
	LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王
	(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,
	如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”
	不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上
	面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,
	要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0*/
	
	
/*第一个方法：采用一个dp方式来判断是否出现两个一样的，然后将五个数来判断最大值和最小值是否小于5，如果小于5的话，那这五个肯定是连续的
 * 反之则不为连续的；
 * 	*/
	public static boolean isContinuous(int[] nums){
		int[] dp=new int[14];
		dp[0]=-5;
		int len=nums.length;
		int max=-1,min=14;
		for(int i=0;i<len;i++){
			dp[nums[i]]++;
			if(nums[i]==0) continue;
			if(dp[nums[i]]>1) return false;
			if(nums[i]>max) max=nums[i];
			if(nums[i]<min) min=nums[i];
		}
		if(max-min<5) return true; else return false;
	}
	
	
//	第二种方法，首先排序以后，来判断数组中0的个数，然后把除0之外的数据进行前后差总和，如果差的总和小于等于0的个数，则说明可以构成连续的数组
//	反之，则不行
	public static boolean IsContinuous(int[] nums) {
		if(nums==null || nums.length<5){
			return false;
		}
		Arrays.sort(nums);
		int numberOfzero=0;
		int numberGap=0;
		for(int i=0;i<nums.length && nums[i]==0;i++){
			numberOfzero++;
		}
		
		if(numberOfzero>2){
			return false;
		}
		int small=numberOfzero;
		int big=small+1;
		while(big<nums.length){
			if(nums[small]==nums[big]){
				return false;
			}
			numberGap+=(nums[big]-nums[small]-1);
			small=big;
			big++;
		}
		return numberGap<=numberOfzero;
	}
	public static void main(String[] args) {
		int[] in={1,3,2,4,5};
		System.out.println(IsContinuous(in));
	}
}
