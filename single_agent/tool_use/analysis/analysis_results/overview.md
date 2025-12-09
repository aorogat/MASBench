# StableToolBench Dataset Analysis

This analysis summarizes the solvable queries and tool environment.

## Query Statistics

- Total solvable queries: **1530**
- Average query length: **43.92 words**
- Min/Max query length: **14 / 155**
- Query groups: `{'test_instruction': 765, 'test_query_ids': 765}`
- API categories (from queries): `{'Data': 272, 'eCommerce': 67, 'Sports': 183, 'Mapping': 30, 'Finance': 275, 'News_Media': 88, 'Other': 76, 'Food': 35, 'Business': 96, 'Artificial_Intelligence_Machine_Learning': 21, 'Financial': 47, 'Media': 111, 'Health_and_Fitness': 48, 'Business_Software': 23, 'Location': 59, 'Search': 42, 'Entertainment': 213, 'Weather': 54, 'Gaming': 52, 'Music': 47, 'Social': 152, 'Education': 59, 'Translation': 16, 'Medical': 21, 'Travel': 53, 'Communication': 36, 'Database': 27, 'Monitoring': 26, 'Text_Analysis': 10, 'Visual_Recognition': 1, 'Logistics': 78, 'Movies': 413, 'Video_Images': 244, 'Tools': 746, 'Devices': 74, 'Cryptography': 13, 'Email': 20, 'Science': 43, 'Advertising': 55, 'SMS': 17, 'Energy': 23, 'Jobs': 26, 'Cybersecurity': 11, 'Transportation': 19, 'Payments': 26, 'Reward': 15, 'Events': 7}`

## Tool Environment Statistics

- Total tools: **658**
- Tool categories: `{'Unknown': 658}`
- Total API endpoints: **1585**
- Avg APIs per tool: **2.41**
- Avg parameters per API: **1.80**

## LaTeX Figure Snippets

Add these directly to your paper:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{query_length_hist.pdf}
  \caption{Query length distribution in StableToolBench.}
\end{figure}
```

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{api_category_bar.pdf}
  \caption{Top API categories used in solvable queries.}
\end{figure}
```

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{apis_per_tool_hist.pdf}
  \caption{Distribution of API endpoints per tool.}
\end{figure}
```
