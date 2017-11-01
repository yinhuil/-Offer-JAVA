package JianZhiOffer;

import java.util.Arrays;
import java.util.Comparator;
import Arrays.LargestRectangleinHistogram;

public class PrintMinNumber33 {
	
	
/*	输入一个正整数数组，把数组里所有数字拼接起来排成一个数，
	打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，
	则打印出这三个数字能排成的最小数字为321323。	*/
	
	
	
//第一种思路，采用先将数组排序，然后使用dp的方法来算	
	public static String PrintMinNumber(int [] numbers) {
    	Arrays.sort(numbers);
    	String[] dp=new String[numbers.length];
    	dp[0]=""+numbers[0];
    	for(int i=1;i<numbers.length;i++){
    		String temp=""+numbers[i];
    		if(compare(dp[i-1], temp)){
    			dp[i]=temp+dp[i-1];
    		}else{
    			dp[i]=dp[i-1]+temp;
    		}
    	}
    	return dp[dp.length-1];
    }
    public static boolean compare(String str1,String str2){
    	int result=(str1+str2).compareTo(str2+str1);
    	if(result>0){
    		return true;
    	}else{
    		return false;
    	}
    }
	
	
	
	
//直接将数组进行排序，其中改变compare的方法；
	
	public static String largestNumber(int[] nums) {
		if(nums==null || nums.length==0){
			return "";
		}
		String[] str=new String[nums.length];
		for(int i=0;i<nums.length;i++){
			str[i]=String.valueOf(nums[i]);
		}
		Comparator<String> comp=new Comparator<String>() {

			@Override
			public int compare(String str1, String str2) {
				String s1=str1+str2;
				String s2=str2+str1;
				return s2.compareTo(s1);
			}
			
		};
		Arrays.sort(str, comp);
		if(str[0].charAt(0)=='0'){
			return "0";
		}
		StringBuilder sb=new StringBuilder();
		for (String string : str) {
			sb.append(string);
		}
		return sb.toString();
	}
	public static void main(String[] args) {
//		int[] in={3,30,34,5,9};
//		System.out.println(largestNumber(in));
		
	}
}
