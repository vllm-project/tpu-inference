<table>
  <thead>
    <tr>
      <th>Checkpoint dtype</th>
      <th>Method</th>
      <th>Supported<br>Hardware Acceleration</th>
      <th>Flax</th>
      <th>Torchax</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AWQ INT4</td>
      <td></td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP4 W4A16</td>
      <td>mxfp4</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>FP8 W8A16</td>
      <td>compressed-tensor</td>
      <td>v7</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>INT4 W4A16</td>
      <td>awq</td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
    <tr>
      <td>INT8 W8A8</td>
      <td>compressed-tensor</td>
      <td>v5, v6</td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
      <td><span title="❓&nbsp;Untested">❓</span></td>
    </tr>
  </tbody>
</table>

> **Note:**
> - *This table only tests checkpoint loading compatibility.*
