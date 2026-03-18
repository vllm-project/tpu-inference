# Update Logic and Layout of Parallelism Techniques Table

This plan outlines how we'll update the `README.md` Parallelism Techniques table based on your latest layout requirement (modeling the `Single-host` and `Multi-host` columns!).

## Branch Confirmation
Yes, **you are on your own new branch** called `parallelism-techniques-update`. We are **not** modifying Rob's branch, so your changes are safe.

## Proposed Changes

We have modified `scripts/update_readme.py` with the following new logic:

### 1. Data Mapping Update
We retained the `CP, DP, EP, PP, SP, TP` rows and successfully re-attached the "multi-host" and "Single-Host-P-D-disaggregation" features as their own individual rows at the bottom! Instead of merging `CorrectnessTest` and `PerformanceTest` into a single status token, the script now cleanly separates them and maps them respectively into the **Single-host** and **Multi-host** fields for each framework!

### 2. New Table Generator (`generate_html_parallelism_table`)
We explicitly structured the custom HTML table to exactly mirror the headers you provided in your screenshot.

Table layout preview showing how it maps over the data:

<table>
  <thead>
    <tr>
      <th rowspan="2">Feature</th>
      <th colspan="2">Flax</th>
      <th colspan="2">torchax</th>
    </tr>
    <tr>
      <th>Single-host</th>
      <th>Multi-host</th>
      <th>Single-host</th>
      <th>Multi-host</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>CP</strong></td>
      <td>unverified</td>
      <td>unverified</td>
      <td>unverified</td>
      <td>unverified</td>
    </tr>
    <tr>
      <td><strong>DP</strong></td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><strong>EP</strong></td>
      <td>✅</td>
      <td>✅ v6e<br>❌ v7x</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><strong>multi-host</strong></td>
      <td>N/A v6e<br>✅ v7x</td>
      <td>N/A v6e<br>unverified v7x</td>
      <td>N/A v6e<br>✅ v7x</td>
      <td>N/A v6e<br>unverified v7x</td>
    </tr>
    <tr>
      <td><strong>PP</strong></td>
      <td>unverified</td>
      <td>unverified</td>
      <td>unverified</td>
      <td>unverified</td>
    </tr>
    <tr>
      <td><strong>Single-Host-P-D-disaggregation</strong></td>
      <td>✅</td>
      <td>N/A</td>
      <td>❌</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td><strong>SP</strong></td>
      <td>❌</td>
      <td>N/A</td>
      <td>❌</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td><strong>TP</strong></td>
      <td>✅</td>
      <td>❌ v6e<br>✅ v7x</td>
      <td>✅</td>
      <td>✅ v6e<br>❌ v7x</td>
    </tr>
  </tbody>
</table>

*(Note: v6e and v7x hardware differences precisely stack inside the `Single-host` and `Multi-host` sub-columns as you expect).*
