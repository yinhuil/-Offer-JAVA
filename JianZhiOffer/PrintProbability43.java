package JianZhiOffer;
//这个没有看  需要继续研究
public class PrintProbability43 {
	private final static int dice=6;
	public static void printDiceProb(int n) {
		if(n<1) return;
		int togger=0;
		int[][] prob=new int[2][];
		prob[0]=new int[dice*n+1];
		prob[1]=new int[dice*n+1];
		
		for(int i=1;i<=dice;i++){
			prob[togger][i]=1;
		}
		for(int k=2;k<=n;k++){
			for(int i=k;i<=k*dice;i++){
				prob[1-togger][i]=0;
				for(int j=1;j<=i && j<=dice;j++){
					prob[1-togger][i]+=prob[togger][i-j];
				}
			}
			togger=1-togger;
		}
		long total=(long) Math.pow(dice, n);
		for(int i=n;i<=dice*n;i++){
			double ratio=(double)prob[togger][i]/total;
			System.out.println(i+"--->"+ratio);
		}
	}

	public static void main(String[] args) {
		printDiceProb(2);
	}
}
