package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.List;

/*Find all possible combinations of k numbers that
add up to a number n, given that only numbers from 1 
to 9 can be used and each combination should be a unique set of numbers.


Example 1:

Input: k = 3, n = 7

Output:

[[1,2,4]]

Example 2:

Input: k = 3, n = 9

Output:

[[1,2,6], [1,3,5], [2,3,4]]*/
public class combinationSum3 {
	public static List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> result=new ArrayList<List<Integer>>();
		List<Integer> list=new ArrayList<Integer>();
		dfs(result,list,k,n,1);
		return result;
	}

	private static void dfs(List<List<Integer>> result, List<Integer> list, int k,
			int n, int start) {
		if(k<0 ||n<0) return;
		else if(k==0 && n==0){
			result.add(new ArrayList<Integer>(list));
		}else{
			for(int i=start;i<=9;i++){
				list.add(i);
				dfs(result, list, k-1, n-i, i+1);
				list.remove(list.size()-1);
			}
		}
		
	}
	public static void main(String[] args) {
		List<List<Integer>> result=combinationSum3(3, 9);
		for (List<Integer> list : result) {
			System.out.println(list);
		}
	}
}























