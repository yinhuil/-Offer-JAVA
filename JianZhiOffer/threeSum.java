package JianZhiOffer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
//在一个数组中找到三个数之和为0的集合。
//思路：   将数组进行排序，   先确定第一个值target，然后再用二分法来让最后两个值的和为-target   （注意：后来一定要把相同的数给跳过去 以防止重复）
public class threeSum {
	public static List<List<Integer>> threeSum(int[] nums) {
	    List<List<Integer>> res = new ArrayList<List<Integer>>();
	    Arrays.sort(nums);
	    for (int i = 0; i + 2 < nums.length; i++) {
	        if (i > 0 && nums[i] == nums[i - 1]) {              // skip same result
	            continue;
	        }
	        int j = i + 1, k = nums.length - 1;  
	        int target = -nums[i];
	        while (j < k) {
	            if (nums[j] + nums[k] == target) {
	                res.add(Arrays.asList(nums[i], nums[j], nums[k]));
	                j++;
	                k--;
	                while (j < k && nums[j] == nums[j - 1]) j++;  // skip same result
	                while (j < k && nums[k] == nums[k + 1]) k--;  // skip same result
	            } else if (nums[j] + nums[k] > target) {
	                k--;
	            } else {
	                j++;
	            }
	        }
	    }
	    return res;
	}
	public static void main(String[] args) {
		int[] in={-1, 0, 1, 2, -1, -4};
		List<List<Integer>> res=threeSum(in);
		for (List<Integer> list : res) {
			System.out.println(list);
		}
	}
}
