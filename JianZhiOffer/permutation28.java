package JianZhiOffer;

import java.util.ArrayList;
import java.util.Collections;
/*题目描述
输入一个字符串,按字典序打印出该字符串中字符的所有排列。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来
的所有字符串abc,acb,bac,bca,cab和cba。*/


public class permutation28 {
	
	
	public static ArrayList<String> permutation(String str){
		ArrayList<String> res=new ArrayList<String>();
		if(str!=null && str.length()>0){
			Helper(str.toCharArray(),0,res);
			Collections.sort(res);
		}
		return res;
	}

	private static void Helper(char[] charArray, int i, ArrayList<String> res) {
		if(i==charArray.length-1){
			String val=String.valueOf(charArray);
			if(!res.contains(val)){
				res.add(val);
			}
		}else{
			for(int j=i;j<charArray.length;j++){
				swap(charArray,i,j);
				Helper(charArray, i+1, res);
				swap(charArray,i,j);
			}
		}
		
	}

	private static void swap(char[] charArray, int i, int j) {
		Character temp=charArray[i];
		charArray[i]=charArray[j];
		charArray[j]=temp;
		
	}

	public static void main(String[] args) {
		String str="abc";
		ArrayList<String> list=permutation(str);
		for (String string : list) {
			System.out.println(string);
		}
	}
}
