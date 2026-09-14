# Security

Storyteller is an experimental local application. It is not a public multi-tenant
service. The MCP process runs with the operator's filesystem permissions.
Only connect trusted clients; isolate campaigns and operating-system accounts
when different people should not share access.

The narrator exposes a fixed tool surface, keeps engine-only settlement and
ruling tools away from the model, and checks durable outcomes before delivery.
These controls reduce failure modes; they do not prove arbitrary model prose
correct or eliminate prompt injection. See [limitations](DEFERRED.md).

The default model endpoint is local HTTP. If you use a remote endpoint, campaign
content and player messages are sent to it. Choose transport and access controls
appropriate to that service. Never expose an unauthenticated inference endpoint
directly to the internet.

Terminal diagnostics are enabled by default. They can contain player messages,
narration, campaign snapshots, and audit records; stderr logs can contain provider
error details. Directories use mode 0700 and files 0600 on supported systems.
`--no-diagnostic-transcript` disables transcript and snapshot recording but not
the stderr log. Campaign backups need the same privacy protection.

Report a suspected vulnerability privately through the repository's GitHub
"Report a vulnerability" facility. If it is unavailable, ask the maintainer for
a private reporting channel without posting exploit details or private data
in a public issue. Include the version, a minimal synthetic reproduction,
expected behavior, and actual behavior. Only the latest release is maintained.
