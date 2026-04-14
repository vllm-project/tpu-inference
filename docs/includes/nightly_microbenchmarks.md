<table>
  <thead>
    <tr>
      <th width="150" style="text-align:left">Category</th>
      <th width="300" style="text-align:left">Test</th>
      <th>W16A16</th>
      <th>W8A8</th>
      <th>W8A16</th>
      <th>W4A4</th>
      <th>W4A8</th>
      <th>W4A16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Moe</b></td>
      <td>Fused&nbsp;MoE</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>gmm</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"><b>Dense</b></td>
      <td>All&#8209;gather&nbsp;matmul</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3"><b>Attention</b></td>
      <td>Generic&nbsp;Ragged&nbsp;Paged<br>Attention&nbsp;V3*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>MLA</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>Ragged&nbsp;Paged<br>Attention&nbsp;V3&nbsp;Head_Dim<br>64*</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> - *For attention kernels, W[x]A[y] denotes KV cache as W, A as compute, and x, y as bit precision.*
