# Dependencies and licensing

Project contributions use GPL-3.0-only. Upstream packages keep their own licenses
and are installed separately. Preserve their license files and any required
notices if distributing an environment, executable, container, or vendored copy.

| Direct component | License |
| --- | --- |
| MCP Python SDK | MIT |
| Pydantic | MIT |
| PyYAML | MIT |
| filelock | MIT |
| prompt-toolkit | BSD-3-Clause |
| Strands Agents | Apache-2.0 |
| OpenAI Python client | Apache-2.0 |
| pytest | MIT |
| pytest-asyncio | Apache-2.0 |
| Ruff | MIT |
| Hatchling (build backend) | MIT |

MIT, BSD-3-Clause, and Apache-2.0 dependencies are compatible with GPLv3.
This does not relicense their upstream distributions.
See the [FSF license list](https://www.gnu.org/licenses/license-list.html.en).

`uv.lock` records the project environment. `requirements-narrator.txt`
records the separate narrator environment, including its transitive pins.
Installed distributions carry their own metadata and license texts. Update
both environments deliberately and review new dependency licenses.

The initial locked environments also include permissively licensed transitive
packages (MIT, BSD, ISC, and Python Software Foundation terms), plus `certifi`
(MPL-2.0) and `tqdm` (MPL-2.0 and MIT). These do not introduce an identified GPLv3
conflict. MPL-covered files retain their own license and source-availability
obligations if redistributed. No installed environments or dependency binaries
are included in this source release.

The Black Sword Hack SRD is CC BY 4.0, covered separately by [NOTICE](../NOTICE).
Gemma 4 and vLLM run as a separately installed service; neither is bundled.
See [model setup](model-setup.md).
