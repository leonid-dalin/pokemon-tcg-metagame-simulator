def resolve_job_id(link_value: str | bytes | None, fallback: str) -> str:
    if isinstance(link_value, bytes):
        try:
            link_value = link_value.decode("utf-8")
        except UnicodeDecodeError:
            return fallback

    if not link_value:
        return fallback

    return link_value
