# Model endpoint

Storyteller connects to a separately managed model server. The configured model is
`google/gemma-4-26B-A4B-it` and the default endpoint is
`http://localhost:8000/v1`. Model weights are not included.

The integration expects chat completions with tool calling, streamed responses,
and strict structured output for classification and settlement. The current
thinking controls send Gemma/vLLM-specific request options. Another model or
provider needs compatibility testing; an OpenAI-compatible URL alone is not
a guarantee of support.

For vLLM, configure the matching Gemma tool and reasoning parsers, automatic
tool choice, and enough context for the system prompt, tools, canon, and conversation.
Optional prefix-cache measurements use the server's `/metrics` endpoint and
per-request prompt-token details. Consult the serving software's own documentation
for the installed version and hardware. The project does not install, start, or
restart an inference server.

The terminal launcher reads `BSH_LLM_BASE_URL` and `BSH_LLM_MODEL`.
The direct narrator CLI accepts `--base-url` and `--model`:

```bash
export BSH_LLM_BASE_URL=http://localhost:8000/v1
export BSH_LLM_MODEL=google/gemma-4-26B-A4B-it
```

The current client is intended for a local endpoint; remote authentication is
not a documented deployment mode. Keep the endpoint private and avoid sending
campaign data to services you do not trust.

Google publishes Gemma 4 under [Apache 2.0](https://ai.google.dev/gemma/apache_2);
the [older Gemma terms](https://ai.google.dev/gemma/terms) explicitly direct
Gemma 4 users to that license. Obtain your selected weights from the publisher
and preserve its notices if you redistribute them. This source release includes
neither weights nor a serving image.
